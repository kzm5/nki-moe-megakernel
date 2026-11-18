"""T8 statistics from ~/runs/p1_bench_<C>_s<S>_r<R>/benchmark_report.json.

Computes per config and seq: TKG p50 / p99 mean ± std (sample), range; pooled std of C and E; split of C's lead
over B into attention (B - E) and fusion (E - C); decision rule |C - E| < 2 x pooled std.

Usage:
    bench_stats.py [--seqs 768 2048 4096] [--out results/phase1/bench_bce.json]
"""
import argparse, glob, json, os, re, statistics as st

H = os.path.expanduser("~")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", type=int, default=[768, 2048, 4096])
    ap.add_argument("--out", default="results/phase1/bench_bce.json")
    a = ap.parse_args()
    res = {}
    for s in a.seqs:
        r = {}
        for c in ("B", "C", "E"):
            p50, p99, e2e = [], [], []
            for f in sorted(glob.glob(f"{H}/runs/p1_bench_{c}_s{s}_r*/benchmark_report.json")):
                d = json.load(open(f))
                p50.append(d["token_generation_model"]["latency_ms_p50"])
                p99.append(d["token_generation_model"]["latency_ms_p99"])
                e2e.append(d["e2e_model"]["latency_ms_p50"])
            if not p50:
                continue
            sd = lambda x: st.stdev(x) if len(x) > 1 else 0.0
            r[c] = {"n": len(p50), "p50": p50, "p50_mean": st.mean(p50), "p50_std": sd(p50), "p50_range": [min(p50), max(p50)],
                    "p99_mean": st.mean(p99), "p99_std": sd(p99), "p99_range": [min(p99), max(p99)],
                    "e2e_p50_mean_ms": st.mean(e2e)}
        if all(k in r for k in "BCE"):
            B, C, E = (r[k]["p50_mean"] for k in "BCE")
            pooled = ((r["C"]["p50_std"] ** 2 + r["E"]["p50_std"] ** 2) / 2) ** 0.5
            r["split"] = {"lead_B_minus_C_ms": B - C, "attention_B_minus_E_ms": B - E, "fusion_E_minus_C_ms": E - C,
                          "pooled_std_CE_ms": pooled, "C_minus_E_lt_2pooled": abs(C - E) < 2 * pooled}
        res[str(s)] = r
        print(s, {k: (round(v["p50_mean"], 3), round(v["p50_std"], 3), v["n"]) for k, v in r.items() if k in "BCE"},
              {k: round(v, 3) if isinstance(v, float) else v for k, v in r.get("split", {}).items()})
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
