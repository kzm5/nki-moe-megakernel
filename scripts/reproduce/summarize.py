"""Summarize reproduce.sh benchmark runs and compare each config/length with the ledger's runs of the same backend.

Computes:
    per config and length: TKG p50 / p99 mean ± std over rounds; the ledger's mean ± std and n for the same
    backend and seq_len (status ok); the difference in ms.

Usage:
    summarize.py BENCH_DIR LEDGER.jsonl
"""
import glob, json, re, statistics as st, sys

bench, ledger = sys.argv[1:3]
runs = {}
for f in sorted(glob.glob(f"{bench}/*_s*_r*/benchmark_report.json")):
    c, s = re.search(r"/([A-Za-z]+)_s(\d+)_r\d+/", f).groups()
    d = json.load(open(f))["token_generation_model"]
    runs.setdefault((c, int(s)), []).append((d["latency_ms_p50"], d["latency_ms_p99"]))
led = {}
for line in open(ledger):
    e = json.loads(line)
    cfg = e.get("config") or {}
    if e.get("status") == "ok" and e.get("tkg_latency_ms_p50") and cfg.get("backend"):
        led.setdefault((cfg["backend"], cfg.get("seq_len")), []).append(e["tkg_latency_ms_p50"])
sd = lambda x: st.stdev(x) if len(x) > 1 else 0.0
out = {}
for (c, s), v in sorted(runs.items()):
    p50 = [x for x, _ in v]
    ref = led.get((c, s), [])
    out[f"{c}_s{s}"] = {"n": len(v), "p50_mean": st.mean(p50), "p50_std": sd(p50), "p99_mean": st.mean([y for _, y in v]),
                        "ledger_n": len(ref), "ledger_p50_mean": st.mean(ref) if ref else None,
                        "ledger_p50_std": sd(ref) if ref else None,
                        "delta_vs_ledger_ms": st.mean(p50) - st.mean(ref) if ref else None}
json.dump(out, open(f"{bench}/summary.json", "w"), indent=1)
for k, v in out.items():
    lp = f"{v['ledger_p50_mean']:.3f} ± {v['ledger_p50_std']:.3f} (n={v['ledger_n']})" if v["ledger_n"] else "–"
    dl = f"{v['delta_vs_ledger_ms']:+.3f}" if v["ledger_n"] else "–"
    print(f"SUMMARY {k}: p50 {v['p50_mean']:.3f} ± {v['p50_std']:.3f} (n={v['n']}); ledger {lp}; delta {dl} ms")
