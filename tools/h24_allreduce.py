"""H2.4 emulated in the CPU golden: precision of the TP=4 all-reduce that sums the o_proj and MoE down-projection partials.

Computes (everything fp32 except the two per-layer reductions; partials split as NxDI shards at TP=4: o_proj K by
head groups, MoE down K by intermediate shards, each partial accumulated in fp32):
    fp32      partials summed in fp32, no rounding (control: must match the golden)
    ar_bf16   partials rounded to bf16, summed in rank order 0..3 with a bf16 rounding after each add (C)
    ag_fp32   partials rounded to bf16, summed in fp32, one rounding to bf16 (V)
Against the frozen golden: KL(golden || variant) per corpus position, top-1 agreement, residual rel-L2 per layer
(same metrics and positions as h22_accum.py).

Usage:
    h24_allreduce.py --variants fp32 ar_bf16 ag_fp32 --out results/phase1/h24.json
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F
from transformers.models.qwen3_moe import modeling_qwen3_moe as M

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h22_accum as H  # noqa: E402

TP = 4
MODE = {"v": "fp32"}


def reduce(parts):
    if MODE["v"] == "fp32":
        return sum(parts)
    if MODE["v"] == "ar_bf16":
        acc = parts[0].bfloat16()
        for p in parts[1:]:
            acc = (acc.float() + p.bfloat16().float()).bfloat16()
        return acc.float()
    return sum(p.bfloat16().float() for p in parts).bfloat16().float()


class TPLinear(torch.nn.Module):
    """Row-parallel linear: TP partials over contiguous K shards, then reduce()."""

    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, x):
        k = self.w.shape[1] // TP
        return reduce([x[..., r * k:(r + 1) * k] @ self.w[:, r * k:(r + 1) * k].T for r in range(TP)])


def moe_forward(self, hidden_states):
    b, s, d = hidden_states.shape
    x = hidden_states.view(-1, d)
    rw = F.softmax(self.gate(x), dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, self.top_k, dim=-1)
    if self.norm_topk_prob:
        rw /= rw.sum(dim=-1, keepdim=True)
    parts = [torch.zeros_like(x) for _ in range(TP)]
    mask = F.one_hot(sel, num_classes=self.num_experts).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        ex = self.experts[e]
        idx, top_x = torch.where(mask[e].squeeze(0))
        xs = x[top_x]
        h = ex.act_fn(ex.gate_proj(xs)) * ex.up_proj(xs)
        wd, k = ex.down_proj.weight, h.shape[-1] // TP
        for r in range(TP):
            parts[r].index_add_(0, top_x, (h[:, r * k:(r + 1) * k] @ wd[:, r * k:(r + 1) * k].T) * rw[top_x, idx, None])
    return reduce(parts).reshape(b, s, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["fp32", "ar_bf16", "ag_fp32"])
    ap.add_argument("--model-path", default=os.path.expanduser("~/qwen-30b-a3b/hf_model"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M.Qwen3MoeSparseMoeBlock.forward = moe_forward
    build = H.G.build

    def build_tp(cls, sd, *args):
        m = build(cls, sd, *args)
        if cls is M.Qwen3MoeDecoderLayer:
            m.self_attn.o_proj = TPLinear(m.self_attn.o_proj.weight)
        return m

    H.G.build = build_tp
    corpus = json.load(open(H.V.CORPUS))
    res = {"tp": TP, "results": []}
    for v in a.variants:
        MODE["v"] = v
        r = H.run("fp32", corpus, a.model_path)
        r["variant"] = v
        res["results"].append(r)
        print("H24", json.dumps({k: r[k] for k in ("variant", "kl_mean", "kl_max", "top1_golden")}),
              "resid L0/L24/L47", [round(r["resid_rel_l2"][L], 6) for L in (0, 24, 47)], flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
