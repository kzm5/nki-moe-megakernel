"""Inspect gross positions from validator captures: per-position KL, top tokens, per-layer router flips and golden margins.

Usage:
    gross_inspect.py PROMPT:POS BUILD=DIR [BUILD=DIR ...] [--base-logits DIR]
"""
import argparse, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validator"))
import validate as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("builds", nargs="+")
    ap.add_argument("--base-logits", default=f"{V.P1}/val/xla_plain")
    a = ap.parse_args()
    pid, pos = a.target.split(":")
    nn, pos = pid[1:], int(pos)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/qwen-30b-a3b/hf_model"))
    g = torch.load(f"{V.GOLDEN}/golden_p{nn}.pt")
    gl = g["logits"][pos]
    gi, gt = g["router_ids"][:, pos].long(), g["router_top"][:, pos].float()
    gap = gt[:, 7] - gt[:, 8]
    e = torch.load(f"{V.REFS}/expected_p{nn}.pt")
    plen = e["input_ids"].shape[1]
    ctx = tok.decode(e["sequences"][0, max(0, plen + pos - 12):plen + pos])
    print(f"{pid}@{pos} kv position {plen - 1 + pos}; context ...{ctx!r}")
    print("golden top5", [(tok.decode([i]), round(float(v), 3)) for v, i in zip(*F.softmax(gl, -1).topk(5))])
    print("golden 8th-9th router gap: min %.5f (L%d), layers with gap < 0.005: %s" % (
        float(gap.min()), int(gap.argmin()), [L for L in range(48) if gap[L] < 0.005]))
    for spec in [f"xla_plain={a.base_logits}"] + a.builds:
        name, d = spec.split("=", 1)
        sc = torch.load(f"{d}/tf/tf_p{nn}.pt")["scores"][pos].float()
        kl = float(V.kl(gl[None], sc[None])[0])
        top = [(tok.decode([i]), round(float(v), 3)) for v, i in zip(*F.softmax(sc, -1).topk(3))]
        line = f"{name:10s} KL {kl:.4f} top3 {top}"
        cp = f"{d}/cap/cap_p{nn}.pt"
        if os.path.exists(cp):
            c = torch.load(cp)
            if c["valid"][pos]:
                lg, ei = c["router_logits"][pos], c["expert_index"]
                fl = []
                for L in range(48):
                    gset = gi[L, :8]
                    if ei is None:
                        f = not V.valid_top8(lg[L], gset)
                    else:
                        ids = ei[pos, L].long()
                        f = not (set(ids.tolist()) == set(gset.tolist()) or (V.valid_top8(lg[L], ids) and V.valid_top8(lg[L], gset)))
                    if f:
                        fl.append(L)
                line += f" | flipped layers {fl}"
        print(line)


if __name__ == "__main__":
    main()
