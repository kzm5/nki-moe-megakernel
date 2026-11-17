"""Per-layer error budget of a build against fp32 golden, from validator stats (golden.py --stats).

Computes per layer L and boundary b (attn, moe, resid), pooled over positions:
    rel-L2 = sqrt(sum ||cand - gold||^2 / sum ||gold||^2), cosine, max-abs; overall and per TKG bucket
    (bucket of the step's KV position plen - 1 + p; p = 0 is the prefill step, reported as "cte")

Returns:
    OUT.json  {"buckets": [...], "<build>": {"<b>": {"rel_l2": [48], "cosine": [48], "maxabs": [48],
               "by_bucket": {bucket: {"rel_l2": [48], "n": int}}}}}
    OUT.csv   build,boundary,bucket,layer,rel_l2,cosine,maxabs,n

Usage:
    error_budget.py --builds xla=DIR [libmoe=DIR ...] --corpus corpus.json --out PREFIX
"""
import argparse, json
import torch

BUCKETS = [128, 256, 512, 768]
BOUNDARIES = ("attn", "moe", "resid")


def bucket(kvpos, p):
    return "cte" if p == 0 else str(next(b for b in BUCKETS if kvpos < b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plen = {p["id"]: p["plen"] for p in json.load(open(a.corpus))["prompts"]}
    res = {"buckets": ["cte"] + [str(b) for b in BUCKETS]}
    lines = ["build,boundary,bucket,layer,rel_l2,cosine,maxabs,n"]
    for spec in a.builds:
        name, d = spec.split("=", 1)
        s = torch.load(f"{d}/stats.pt")
        ids = s["prompt_ids"]
        bk = [bucket(plen[ids[j]] - 1 + q, q) for j, q in s["index"]]
        out = {}
        for b in BOUNDARIES:
            e2, g2, dot, c2, mx = (s[f"{b}.{k}"] for k in ("err2", "gold2", "dot", "cand2", "maxabs"))
            ok = ~torch.isnan(e2)

            def agg(m):
                m = m[:, None] & ok
                z = lambda t: torch.where(m, t, torch.zeros_like(t)).sum(0)
                return ((z(e2) / z(g2)).sqrt(), z(dot) / (z(c2).sqrt() * z(g2).sqrt()),
                        torch.where(m, mx, torch.zeros_like(mx)).amax(0), int(m[:, 0].sum()))
            allm = torch.ones(len(bk), dtype=torch.bool)
            rel, cos, mxa, n = agg(allm)
            r = {"rel_l2": [float(x) for x in rel], "cosine": [float(x) for x in cos],
                 "maxabs": [float(x) for x in mxa], "n": n, "by_bucket": {}}
            for L in range(48):
                lines.append(f"{name},{b},all,{L},{rel[L]:.6g},{cos[L]:.8f},{mxa[L]:.6g},{n}")
            for bb in res["buckets"]:
                m = torch.tensor([x == bb for x in bk])
                if not m.any():
                    continue
                rel_b, cos_b, mx_b, nb = agg(m)
                r["by_bucket"][bb] = {"rel_l2": [float(x) for x in rel_b], "n": nb}
                for L in range(48):
                    lines.append(f"{name},{b},{bb},{L},{rel_b[L]:.6g},{cos_b[L]:.8f},{mx_b[L]:.6g},{nb}")
            out[b] = r
        res[name] = out
    json.dump(res, open(f"{a.out}.json", "w"), indent=1)
    open(f"{a.out}.csv", "w").write("\n".join(lines) + "\n")
    for spec in a.builds:
        name = spec.split("=", 1)[0]
        for b in BOUNDARIES:
            r = res[name][b]["rel_l2"]
            print(name, b, "L0 %.4f L23 %.4f L47 %.4f max %.4f (L%d)" % (r[0], r[23], r[47], max(r), r.index(max(r))),
                  {k: v["n"] for k, v in res[name][b]["by_bucket"].items()})


if __name__ == "__main__":
    main()
