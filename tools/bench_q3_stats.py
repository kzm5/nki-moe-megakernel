"""Statistics of a bench_q3.sh session: TKG p50 / p99 mean ± std per config, deltas vs a reference with 2 × pooled std.

Usage:
    bench_q3_stats.py TAG --ref E --out results/phase1/bench_q3_TAG.json
"""
import argparse, glob, json, os, re, statistics as st

H = os.path.expanduser("~")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--ref", nargs="+", default=["E", "C"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for f in sorted(glob.glob(f"{H}/runs/p1_q3_{a.tag}_*_s768_r*/benchmark_report.json")):
        c = re.search(rf"p1_q3_{a.tag}_(.+)_s768_r\d+", f).group(1)
        d = json.load(open(f))["token_generation_model"]
        runs.setdefault(c, []).append((d["latency_ms_p50"], d["latency_ms_p99"]))
    sd = lambda x: st.stdev(x) if len(x) > 1 else 0.0
    res = {}
    for c, v in runs.items():
        p50, p99 = [x for x, _ in v], [y for _, y in v]
        res[c] = {"n": len(v), "p50": p50, "p50_mean": st.mean(p50), "p50_std": sd(p50),
                  "p99_mean": st.mean(p99), "p99_std": sd(p99)}
    for c in res:
        for ref in a.ref:
            if ref in res and ref != c:
                pooled = ((res[c]["p50_std"] ** 2 + res[ref]["p50_std"] ** 2) / 2) ** 0.5
                res[c][f"minus_{ref}_ms"] = res[c]["p50_mean"] - res[ref]["p50_mean"]
                res[c][f"two_pooled_std_{ref}_ms"] = 2 * pooled
    json.dump(res, open(a.out, "w"), indent=1)
    for c, r in res.items():
        print(c, f"{r['p50_mean']:.3f} ± {r['p50_std']:.3f} (n={r['n']})",
              {k: round(v, 3) for k, v in r.items() if k.startswith(("minus", "two"))})


if __name__ == "__main__":
    main()
