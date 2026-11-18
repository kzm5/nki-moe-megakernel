# coding=utf-8
"""Qwen3 MoE, config B plus one null NKI call per decoder layer on TKG (identity copy of the attention output).

Everything else is qwen.py (config B with --moe-fused-nki-kernel-enabled): same weights, MoE path and compiler
flags. The difference to B is the cost of 48 NKI call boundaries per decode step.
"""
import torch

from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

from megakernels.qwen3_moe import qwen
from megakernels.qwen3_moe.null_nki import null_tkg_jit


class NeuronQwen3MoeDecoderLayerNull(qwen.NeuronQwen3MoeDecoderLayer):
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                padding_mask=None, **kwargs):
        if kwargs.get("is_for_context_encoding", True):
            return super().forward(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                                   past_key_value=past_key_value, padding_mask=padding_mask, **kwargs)
        residual = hidden_states
        qkv_fused_rmsnorm = None
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, rmsnorm=qkv_fused_rmsnorm, **kwargs)
        hidden_states = null_tkg_jit[2](hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        if not self.moe_fused_nki_kernel_enabled:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, padding_mask)[0]
        hidden_states = residual + hidden_states
        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class NeuronQwen3MoeModelNull(qwen.NeuronQwen3MoeModel):
    def init_model(self, config):
        super().init_model(config)
        self.layers = torch.nn.ModuleList(
            [NeuronQwen3MoeDecoderLayerNull(config, i) for i in range(config.num_hidden_layers)])


class NeuronQwen3MoeForCausalLM(qwen.NeuronQwen3MoeForCausalLM):
    _model_cls = NeuronQwen3MoeModelNull
