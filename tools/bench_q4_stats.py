"""Statistics of a bench_q4.sh session: TKG p50 / p99 mean ± std per config and length, pairwise deltas with 2 × pooled std,
and the same deltas within each round (paired).

Usage:
    bench_q4_stats.py TAG --pairs Eh-C Eh-B V-C V-B --out results/phase1/bench_TAG.json
"""
import argparse, glob, json, os, re, statistics as st

H = os.path.expanduser("~")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--pairs", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for f in sorted(glob.glob(f"{H}/runs/p1_{a.tag}_*_s*_r*/benchmark_report.json")):
        c, s, r = re.search(rf"p1_{a.tag}_(.+)_s(\d+)_r(\d+)/", f).groups()
        d = json.load(open(f))["token_generation_model"]
        runs.setdefault(int(s), {}).setdefault(c, {})[int(r)] = (d["latency_ms_p50"], d["latency_ms_p99"])
    sd = lambda x: st.stdev(x) if len(x) > 1 else 0.0
    res = {}
    for s, cfgs in sorted(runs.items()):
        out = {"configs": {}, "pairs": {}}
        for c, rr in cfgs.items():
            p50, p99 = [rr[k][0] for k in sorted(rr)], [rr[k][1] for k in sorted(rr)]
            out["configs"][c] = {"n": len(p50), "p50_mean": st.mean(p50), "p50_std": sd(p50), "p99_mean": st.mean(p99),
                                 "p99_std": sd(p99), "p50_runs": p50, "p99_runs": p99, "rounds": sorted(rr)}
        for pr in a.pairs:
            x, y = pr.split("-")
            if x not in cfgs or y not in cfgs:
                continue
            cx, cy = out["configs"][x], out["configs"][y]
            pooled = ((cx["p50_std"] ** 2 + cy["p50_std"] ** 2) / 2) ** 0.5
            common = sorted(set(cfgs[x]) & set(cfgs[y]))
            paired = [cfgs[x][k][0] - cfgs[y][k][0] for k in common]
            out["pairs"][pr] = {"delta_ms": cx["p50_mean"] - cy["p50_mean"], "two_pooled_std": 2 * pooled,
                                "delta_pct_of_ref": 100 * (cx["p50_mean"] - cy["p50_mean"]) / cy["p50_mean"],
                                "paired_deltas": paired, "paired_mean": st.mean(paired), "paired_std": sd(paired)}
        res[str(s)] = out
    json.dump(res, open(a.out, "w"), indent=1)
    for s, o in res.items():
        print(f"seq {s}")
        for c, v in o["configs"].items():
            print(f"  {c:3s} n={v['n']} p50 {v['p50_mean']:.3f} ± {v['p50_std']:.3f}  p99 {v['p99_mean']:.3f} ± {v['p99_std']:.3f}  runs {[round(x, 3) for x in v['p50_runs']]}")
        for pr, v in o["pairs"].items():
            print(f"  {pr:5s} {v['delta_ms']:+.3f} ms (2σ pooled {v['two_pooled_std']:.3f}; {v['delta_pct_of_ref']:+.2f}%)  paired {[round(x, 3) for x in v['paired_deltas']]} mean {v['paired_mean']:+.3f} ± {v['paired_std']:.3f}")


if __name__ == "__main__":
    main()
