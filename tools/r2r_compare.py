"""Run-to-run comparison of two teacher-forced device captures of the same build, against the golden.

Computes, over all corpus positions: max-abs logit difference, KL(run1 || run2), top-1 changes between runs,
fraction of differing logits and of differing positions; the same per prompt; and each run's mean KL vs golden.

Usage:
    r2r_compare.py RUN1_DIR RUN2_DIR --out OUT.json
"""
import argparse, glob, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validator"))
import validate as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kls, mx, t1, ndiff, nall, pos_diff, npos, kg = [], 0.0, 0, 0, 0, 0, 0, {"a": [], "b": []}
    per = {}
    for f in sorted(glob.glob(f"{a.a}/tf/tf_p*.pt")):
        nn = os.path.basename(f)[4:-3]
        x = torch.load(f)["scores"].float()
        y = torch.load(f"{a.b}/tf/tf_p{nn}.pt")["scores"].float()
        g = torch.load(f"{V.GOLDEN}/golden_p{nn}.pt")["logits"]
        fin = torch.isfinite(x) & torch.isfinite(y)
        d = (x - y).abs().masked_fill(~fin, 0)
        k = V.kl(x, y)
        kls.append(k)
        mx = max(mx, float(d.max()))
        ch = int((x.argmax(-1) != y.argmax(-1)).sum())
        t1 += ch
        ndiff += int((d > 0).sum()); nall += int(fin.sum())
        pd = int((d.amax(-1) > 0).sum()); pos_diff += pd; npos += x.shape[0]
        kg["a"].append(V.kl(g, x)); kg["b"].append(V.kl(g, y))
        per[nn] = {"max_abs": float(d.max()), "kl_mean": float(k.mean()), "top1_changes": ch, "positions_differ": pd,
                   "n": x.shape[0]}
    k = torch.cat(kls)
    res = {"n_positions": npos, "positions_differ": pos_diff, "logits_differ_frac": ndiff / nall,
           "max_abs_logit_diff": mx, "kl_run1_run2_mean": float(k.mean()), "kl_run1_run2_max": float(k.max()),
           "kl_run1_run2_p99": float(k.quantile(0.99)), "top1_changes": t1,
           "kl_vs_golden_mean": {s: float(torch.cat(v).mean()) for s, v in kg.items()}, "per_prompt": per}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}))


if __name__ == "__main__":
    main()
