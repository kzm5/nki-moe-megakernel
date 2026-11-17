"""Torch-level faults injected into the XLA model at trace time (runtime monkeypatches; no library edits).

Mutants:
    m1  expert down-projection accumulated in bf16: the intermediate dim is split into CHUNK-wide partial
        products, each rounded to bf16 and summed in bf16
    m2  top-k affinity renormalization dropped (norm_topk_prob=False)
    m3  control: decode expert weighted sum taken in reverse order, accumulated in fp32, cast once
    m4  at MUTANT_LAYER, the 8th selected expert replaced by the 9th
"""
import torch
import torch.nn.functional as F

CHUNK = 8
MUTANT_LAYER = 24
N_LAYERS = 48


def m1():
    from neuronx_distributed.modules.moe import moe_parallel_layers as mpl
    from neuronx_distributed.parallel_layers import mappings

    def forward(self, input_, expert_indices=None):
        weight = self.weight[expert_indices, :, :] if expert_indices is not None else self.weight
        acc = None
        for k in range(0, weight.shape[1], CHUNK):
            part = torch.einsum("e...h,ehi->e...i", input_[..., k:k + CHUNK], weight[:, k:k + CHUNK, :])
            part = part.to(torch.bfloat16)
            acc = part if acc is None else (acc + part).to(torch.bfloat16)
        out = acc.to(input_.dtype)
        if self.reduce_output:
            out = mappings.reduce_from_tensor_model_parallel_region(out, process_group=self.tensor_parallel_group)
        if self.bias is not None:
            bias = self.bias[expert_indices, :] if expert_indices is not None else self.bias
            out = out + bias.unsqueeze(1).unsqueeze(2)
        return out
    mpl.ExpertFusedRowParallelLinear.forward = forward


def m2():
    from megakernels.qwen3_moe import qwen
    orig = qwen.Qwen3MoeInferenceConfig.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.neuron_config.normalize_top_k_affinities = False
    qwen.Qwen3MoeInferenceConfig.__init__ = init


def m3():
    from neuronx_distributed.modules.moe import expert_mlps_v2 as em

    def forward_selective_loading(self, hidden_states, expert_affinities, expert_index):
        mlp_op = self.get_mlp_op()
        T = hidden_states.shape[0]
        aff = expert_affinities[torch.arange(T, device=hidden_states.device).unsqueeze(1), expert_index]
        if self.routed_experts_mlp_config.normalize_top_k_affinities:
            aff = F.normalize(aff, p=1.0, dim=1)
        outs = []
        for t in range(T):
            y = mlp_op(hidden_states[t].unsqueeze(0).unsqueeze(1), expert_indices=expert_index[t])
            terms = (y.squeeze(1) * aff[t].unsqueeze(1)).float()
            acc = terms[-1]
            for j in range(terms.shape[0] - 2, -1, -1):
                acc = acc + terms[j]
            outs.append(acc.to(hidden_states.dtype))
        return torch.stack(outs, dim=0)
    em.ExpertMLPsV2.forward_selective_loading = forward_selective_loading


def m4():
    from neuronx_distributed.modules.moe import routing
    order = {}

    def forward(self, hidden_states):
        assert not self.apply_act_fn_over_topk
        layer = order.setdefault(id(self), len(order)) % N_LAYERS
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)
        if layer == MUTANT_LAYER:
            _, idx = torch.topk(router_logits, self.top_k + 1)
            idx = torch.cat([idx[:, : self.top_k - 1], idx[:, self.top_k:]], dim=1)
        else:
            _, idx = torch.topk(router_logits, self.top_k)
        return router_logits, expert_affinities.to(dtype=hidden_states.dtype), idx.detach().to(dtype=torch.long)
    routing.RouterTopK.forward = forward


def apply(name):
    {"m1": m1, "m2": m2, "m3": m3, "m4": m4}[name]()
    print(f"MUTANT {name} applied", flush=True)
