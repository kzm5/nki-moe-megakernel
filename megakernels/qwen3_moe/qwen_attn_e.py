# coding=utf-8
"""Qwen3 MoE, config E: library MoE kernel + the megakernel's TKG attention stage, one NKI call per layer.

CTE: stock per-layer path (same as qwen.py).
TKG: per layer, self_attn runs attn_e_tkg (vendored attention_block_tkg with fused input RMSNorm and in-place
     K/V update, then the TP all-reduce); residual adds and the MoE block (library fused MoE kernel when
     --moe-fused-nki-kernel-enabled) stay in the XLA graph.
Weights: qwen.py's conversion plus the megakernel's rank-grouped fused QKV ([H, I_QKV] per rank) and Wo^T.
Compiler flags: qwen.py's, plus the backend-verifier opt-out required by the in-place KV scatter.

Env:
    ATTN_E_HOIST=1  build the attention mask and permuted RoPE tables once per step in the graph and pass them
                    to attn_e_hoist_tkg (default 0: attn_e_tkg builds them in every call)
"""
import os
from typing import Optional, Tuple

import torch

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

from megakernels.qwen3_moe import qwen
from megakernels.qwen3_moe.attn_e_hoist_nki import attn_e_hoist_tkg_jit
from megakernels.qwen3_moe.attn_e_nki import attn_e_tkg_jit
from megakernels.qwen3_moe.qwen_with_megakernel import PreTransposedColumnParallelLinear
from megakernels.qwen3_moe.transformer_qwen3_moe_speculative import D_HEAD, NUM_Q_HEADS_TP

HOIST = os.environ.get("ATTN_E_HOIST", "0") == "1"
_STEP = {}  # per traced step: position_ids -> (mask, cos_perm, sin_perm), shared by all layers


def _step_tensors(position_ids, cos, sin, S_ctx):
    """Same layouts as _build_full_mask_hbm / _build_permuted_rope_hbm_from_pos."""
    if _STEP.get("pid") is not position_ids:
        B, S_tkg = position_ids.shape
        s = torch.arange(S_ctx, device=position_ids.device).view(S_ctx, 1, 1, 1)
        t = torch.arange(S_tkg, device=position_ids.device).view(1, 1, 1, S_tkg)
        base = position_ids[:1, :1].view(1, 1, 1, 1)  # batch 0, as the kernel builder
        active = (s >= S_ctx - S_tkg) & (s <= S_ctx - S_tkg + t)
        mask = ((s < base) | active).expand(S_ctx, B, NUM_Q_HEADS_TP, S_tkg).to(torch.bfloat16).contiguous()
        half = D_HEAD // 2
        _STEP.update(pid=position_ids, t=(mask, cos[..., :half].permute(2, 0, 1).contiguous(),
                                          sin[..., :half].permute(2, 0, 1).contiguous()))
    return _STEP["t"]


class Qwen3MoeENeuronConfig(MoENeuronConfig):
    def __init__(self, **kwargs):
        # Full-KV layout + NxDI skips its KV scatter and update_cache on TKG (kernel writes K/V in place).
        kwargs["attn_tkg_nki_kernel_enabled"] = True
        kwargs["attn_block_tkg_nki_kernel_cache_update"] = True
        kwargs.setdefault("fused_qkv", False)
        super().__init__(**kwargs)


def convert_qwen3_moe_e_state_dict(neuron_state_dict, config):
    tp = config.neuron_config.tp_degree
    for l in range(config.num_hidden_layers):  # noqa: E741
        p = f"layers.{l}.self_attn"
        q, k, v = (neuron_state_dict[f"{p}.{n}_proj.weight"] for n in ("q", "k", "v"))
        qd, kd = q.shape[0] // tp, k.shape[0] // tp
        blocks = []
        for r in range(tp):
            blocks += [q[r * qd:(r + 1) * qd], k[r * kd:(r + 1) * kd], v[r * kd:(r + 1) * kd]]
        neuron_state_dict[f"{p}.Wqkv_nki.weight"] = torch.cat(blocks, dim=0).detach().contiguous()
        neuron_state_dict[f"{p}.Wo_nki.weight"] = neuron_state_dict[f"{p}.o_proj.weight"].T.contiguous()
    return qwen.convert_qwen3_moe_hf_to_neuron_state_dict(neuron_state_dict, config)


class NeuronQwen3MoEAttentionE(qwen.NeuronQwen3MoEAttention):
    def __init__(self, config):
        # Keep NxDI's own TKG attention-block kernel out of this module (the flag is promoted from
        # attn_tkg_nki_kernel_enabled); TKG is dispatched to attn_e_tkg in forward.
        saved = config.neuron_config.attn_block_tkg_nki_kernel_enabled
        config.neuron_config.attn_block_tkg_nki_kernel_enabled = False
        try:
            super().__init__(config)
        finally:
            config.neuron_config.attn_block_tkg_nki_kernel_enabled = saved
        dt = config.neuron_config.torch_dtype
        H = config.hidden_size
        hq, hkv = config.num_attention_heads * config.head_dim, config.num_key_value_heads * config.head_dim
        self.Wqkv_nki = PreTransposedColumnParallelLinear(H, hq + 2 * hkv, bias=False, gather_output=False, dtype=dt)
        self.Wo_nki = ColumnParallelLinear(H, hq, bias=False, gather_output=False, dtype=dt)
        self._replica_groups = (list(range(config.neuron_config.tp_degree)),)
        self.attn_out_id = torch.nn.Identity()  # hook point for tensor capture of the TKG attention output

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, rmsnorm=None,
                e_tkg=None, **kwargs):
        if e_tkg is None:
            return super().forward(hidden_states=hidden_states, attention_mask=attention_mask,
                                   position_ids=position_ids, past_key_value=past_key_value, rmsnorm=rmsnorm,
                                   **kwargs)
        K, V, gpre = e_tkg
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        if HOIST:
            mask, cos_p, sin_p = _step_tensors(position_ids, cos, sin, K.shape[-2])
            Y, K_out, V_out = attn_e_hoist_tkg_jit[2](
                hidden_states, self.Wqkv_nki.weight, self.Wo_nki.weight,
                self.q_layernorm.weight, self.k_layernorm.weight, gpre,
                K, V, mask, cos_p, sin_p, position_ids.to(torch.int32), replica_groups=self._replica_groups)
            return self.attn_out_id(Y), (K_out, V_out), cos, sin
        Y, K_out, V_out = attn_e_tkg_jit[2](
            hidden_states, self.Wqkv_nki.weight, self.Wo_nki.weight,
            self.q_layernorm.weight, self.k_layernorm.weight, gpre,
            K, V, cos, sin, position_ids.to(torch.int32), replica_groups=self._replica_groups)
        return self.attn_out_id(Y), (K_out, V_out), cos, sin


class NeuronQwen3MoeDecoderLayerE(qwen.NeuronQwen3MoeDecoderLayer):
    def __init__(self, config, layer_idx: int):
        saved = config.neuron_config.attn_block_tkg_nki_kernel_enabled
        config.neuron_config.attn_block_tkg_nki_kernel_enabled = False
        try:
            super().__init__(config, layer_idx)
        finally:
            config.neuron_config.attn_block_tkg_nki_kernel_enabled = saved
        self.layer_idx = layer_idx
        self.self_attn = NeuronQwen3MoEAttentionE(config=config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                padding_mask=None, kv_mgr=None, is_for_context_encoding=True, **kwargs):
        if is_for_context_encoding:
            return super().forward(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                                   past_key_value=past_key_value, padding_mask=padding_mask, kv_mgr=kv_mgr,
                                   is_for_context_encoding=is_for_context_encoding, **kwargs)
        assert kv_mgr is not None, "TKG path requires kv_mgr (attn_block_tkg_nki_kernel_cache_update)"
        i = self.layer_idx
        K, V = kv_mgr.past_key_values[2 * i], kv_mgr.past_key_values[2 * i + 1]
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        residual = hidden_states
        attn, kv, cos, sin = self.self_attn(hidden_states, position_ids=position_ids,
                                            e_tkg=(K, V, self.input_layernorm.weight))
        hidden_states = residual + attn
        residual = hidden_states
        if not self.moe_fused_nki_kernel_enabled:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, padding_mask)[0]
        hidden_states = residual + hidden_states
        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, kv, cos, sin, None)


class NeuronQwen3MoeModelE(qwen.NeuronQwen3MoeModel):
    def init_model(self, config):
        super().init_model(config)
        self.layers = torch.nn.ModuleList(
            [NeuronQwen3MoeDecoderLayerE(config, i) for i in range(config.num_hidden_layers)])


class NeuronQwen3MoeForCausalLM(qwen.NeuronQwen3MoeForCausalLM):
    _model_cls = NeuronQwen3MoeModelE

    @classmethod
    def get_neuron_config_cls(cls):
        return Qwen3MoeENeuronConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        return convert_qwen3_moe_e_state_dict(state_dict, config)

    def get_compiler_args(self):
        nc = self.neuron_config
        saved = nc.attn_block_tkg_nki_kernel_enabled
        nc.attn_block_tkg_nki_kernel_enabled = False
        try:
            args = super().get_compiler_args()
        finally:
            nc.attn_block_tkg_nki_kernel_enabled = saved
        return args + " --internal-backend-options=--enable-verifier=false"
