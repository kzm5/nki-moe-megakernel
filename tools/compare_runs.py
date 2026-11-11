"""Run-to-run consistency of saved greedy captures (expected_pNN.pt files).

Computes (per pair of capture dirs, per prompt):
    bit_identical (tokens and logits), first token divergence, and over positions before it:
    rel_l2, max_abs, KL(a || b), top-1 agreement; with --golden also each run's rel_l2 vs golden

Usage:
    compare_runs.py OUT.json NAME=DIR NAME=DIR [NAME=DIR ...] [--golden DIR]
"""
import argparse, itertools, json, os
import torch
import torch.nn.functional as F


def load(d, i):
    e = torch.load(f"{d}/expected_p{i:02d}.pt")
    return e["sequences"][0], e["logits"][:, 0].float()


def stats(a, b):
    keep = torch.isfinite(a).all(0) & torch.isfinite(b).all(0)
    a, b = a[:, keep].double(), b[:, keep].double()
    la, lb = F.log_softmax(a, -1), F.log_softmax(b, -1)
    return {"rel_l2": float(((a - b).norm(dim=-1) / a.norm(dim=-1)).mean()), "max_abs": float((a - b).abs().max()),
            "kl": float((la.exp() * (la - lb)).sum(-1).mean()), "top1": float((a.argmax(-1) == b.argmax(-1)).double().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--golden")
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.runs)
    n = len([f for f in os.listdir(next(iter(runs.values()))) if f.startswith("expected_p")])
    res = {"pairs": {}, "vs_golden": {}}
    for x, y in itertools.combinations(runs, 2):
        rows = []
        for i in range(n):
            (sx, lx), (sy, ly) = load(runs[x], i), load(runs[y], i)
            same_tok = torch.equal(sx, sy)
            row = {"prompt": i, "bit_identical": bool(same_tok and torch.equal(lx, ly)), "tokens_identical": bool(same_tok)}
            if not row["bit_identical"]:
                diff = (sx != sy).nonzero()
                plen = len(sx) - lx.shape[0]
                fd = int(diff[0, 0]) - plen if len(diff) else -1
                m = lx.shape[0] if fd < 0 else fd
                row.update(first_div=fd, n_pos=m, **(stats(lx[:m], ly[:m]) if m else {}))
            rows.append(row)
        res["pairs"][f"{x}|{y}"] = {"n_bit_identical": sum(r["bit_identical"] for r in rows),
                                     "n_tokens_identical": sum(r["tokens_identical"] for r in rows), "per_prompt": rows}
        print(x, y, json.dumps({k: v for k, v in res["pairs"][f"{x}|{y}"].items() if k != "per_prompt"}), flush=True)
    if a.golden:
        for x in runs:
            num = den = 0.0
            for i in range(n):
                g = torch.load(f"{a.golden}/golden_p{i:02d}.pt")["logits"].double()
                _, lx = load(runs[x], i)
                keep = torch.isfinite(lx).all(0)
                num += float(((lx[:, keep].double() - g[:, keep]).norm(dim=-1) / g[:, keep].norm(dim=-1)).sum()); den += lx.shape[0]
            res["vs_golden"][x] = {"rel_l2_mean": num / den}
        print("vs_golden", json.dumps(res["vs_golden"]))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
