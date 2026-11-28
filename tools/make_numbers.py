"""NUMBERS.md: every poster number with its source (ledger run ids, or report file and field).

Usage:
    make_numbers.py --out results/final/NUMBERS.md
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures as F  # noqa: E402

P1 = F.P1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b = F.bench({"q5_t6", "q4b_t10"})
    rows = []
    add = lambda k, v, src: rows.append((k, v, src))
    ids = lambda c, s: ", ".join(f"`{i}`" for i in F.stat(b, c, s)["ids"])
    # latency
    for c in ("A", "B", "C", "E", "Eh", "V"):
        for s in F.SEQS:
            x = F.stat(b, c, s)
            if x:
                add(f"TKG p50 {c} @ {s}", f"{x['p50']:.3f} ± {x['p50_sd']:.3f} ms (n={x['n']}); p99 {x['p99']:.3f} ± {x['p99_sd']:.3f}",
                    f"ledger {ids(c, s)}")
    for x, y in (("C", "B"), ("V", "B"), ("V", "C"), ("Eh", "C"), ("Eh", "B"), ("E", "B"), ("Eh", "E"), ("A", "B")):
        for s in F.SEQS:
            d, ref = F.delta(b, x, y, s), F.stat(b, y, s)
            if d:
                add(f"{x} − {y} @ {s}", f"{d['ms']:+.3f} ms ({100 * d['ms'] / ref['p50']:+.2f}% of {y}; 2σ pooled {d['two_sigma']:.3f})",
                    f"ledger rows of TKG p50 {x} and {y} @ {s} above")
    xa, xb = F.stat(b, "A", 768), F.stat(b, "B", 768)
    add("B speedup over XLA @ 768", f"{xa['p50'] / xb['p50']:.3f}x", "TKG p50 A and B @ 768 above")
    # gates
    for m in ("m1", "m2", "m3", "m4"):
        r = F.J(f"reports_v11/{m}.json")
        add(f"Gate 1: mutant {m} v1.1 verdict", "PASS" if r["pass"] else f"FAIL ({', '.join(r['rows_failed'])})",
            f"`results/phase1/reports_v11/{m}.json` pass / rows_failed")
    for s in F.SEQS:
        v, bb = F.stat(b, "V", s), F.stat(b, "B", s)
        if v and bb:
            add(f"Gate 4: V gain over B @ {s}" + (" (judged)" if s == 768 else " (secondary)"),
                f"{100 * (bb['p50'] - v['p50']) / bb['p50']:.2f}% (threshold 5%)", "TKG p50 V and B above")
    # accuracy
    for k in ("xla", "B", "E", "Eh", "C", "V"):
        r = F.acc_row(k)
        src = f"`results/phase1/{F.ACC_REPORTS[k]}` rows"
        add(f"{r['config']}: v1.1 verdict, KL ratio vs XLA, ppl", f"{r['v1.1']}, {r['kl_ratio_vs_xla']:.4f}, {r['ppl']:.4f}",
            src + " logits_tf.kl_ratio, perplexity.ppl_cand")
        if r["rho_resid"] is not None:
            add(f"{r['config']}: median ρ attn / moe / resid, router flips",
                f"{r['rho_attn']:.3f} / {r['rho_moe']:.3f} / {r['rho_resid']:.3f}, {r['router_flip_pct']:.2f}%",
                src + " op_boundary.per_boundary.*.rho_median, router.flip_pct_cand")
    # determinism
    n_ok = 0
    for f in ("t4/c1_2_r2r.json", "t4/c1_2_r2r3.json", "t4/c1_2_r2r4.json", "t4/c1_2_r2r5.json"):
        d = F.J(f)
        n_ok += d["positions_differ"] == 0 and d["max_abs_logit_diff"] == 0
    add("V cross-process determinism", f"captures 2–5 each bit-identical to capture 1 ({n_ok}/4, 7,264 positions each)",
        "`results/phase1/t4/c1_2_r2r{,3,4,5}.json` positions_differ, max_abs_logit_diff")
    d = F.J("t4/V_t1_vs_c1_2.json")
    add("V + asserts (merged, `a62b8af`) vs V", f"{d['positions_differ']} of {d['n_positions']} positions differ",
        "`results/phase1/t4/V_t1_vs_c1_2.json`")
    # precision studies
    cv = F.J("h24_C_vs_V.json")
    add("H2.4 device: C vs V prompts with any logit difference",
        f"{sum(1 for v in cv['per_prompt'].values() if v['positions_differ'])} of {len(cv['per_prompt'])}; top-1 changes {cv['top1_changes']}",
        "`results/phase1/h24_C_vs_V.json` per_prompt")
    h = {v["variant"]: v for v in F.J("h24.json")["results"]}
    add("H2.4 emulation: bf16-per-add all-reduce vs fp32 sum, mean KL vs golden",
        f"{h['ar_bf16']['kl_mean']:.3e} vs {h['ag_fp32']['kl_mean']:.3e} ({100 * (h['ar_bf16']['kl_mean'] / h['ag_fp32']['kl_mean'] - 1):+.0f}%)",
        "`results/phase1/h24.json` results[].kl_mean")
    h = {v["variant"]: v for v in F.J("h22.json")["results"]}
    add("H2.2 emulation: expert GEMMs bf16 inputs / bf16 accumulation, mean KL vs golden",
        f"{h['bf16in']['kl_mean']:.3e} / {h['bf16acc']['kl_mean']:.3e}", "`results/phase1/h22.json` results[].kl_mean")
    eb = F.J("error_budget/xla_B_E.json")
    r = lambda k, L: eb[k]["resid"]["rel_l2"][L]
    peak = max(range(48), key=lambda L: r("xla", L))
    add("H2.6: residual rel-L2 vs golden at L0 / L47, XLA vs B vs E",
        f"{r('xla', 0):.4f} / {r('xla', 47):.4f}; {r('B', 0):.4f} / {r('B', 47):.4f}; {r('E', 0):.4f} / {r('E', 47):.4f}; "
        f"B at L0 {100 * (1 - r('B', 0) / r('xla', 0)):.0f}% below XLA; XLA peak at L{peak}",
        "`results/phase1/error_budget/xla_B_E.json` [build].resid.rel_l2")
    # long context validation
    for c in ("xla", "B", "C", "V"):
        for f, what in ((f"t4_s2048/{c}.json", "v1 corpus @ 2048"), (f"t4_s4096/{c}.json", "v1 corpus @ 4096"),
                        (f"corpus_long/reports/{c}_s2048.json", "supplementary long corpus @ 2048")):
            r = F.J(f)
            r = r.get("report", r)
            add(f"{F.NAMES.get(c, c)}: validation, {what}", f"{'PASS' if r['pass'] else 'FAIL'}, KL ratio {r['rows']['logits_tf']['kl_ratio']:.4f}",
                f"`results/phase1/{f}`")
    add("C @ 8192 runtime", "hang in the 8192 TKG bucket (execution timeout, FATAL-RT-UNDEFINED-STATE); 1 attempt",
        "ledger `p1_q4b_t10_compile_C_s8192_20261126`; FINDINGS Q4b T10")
    # profile
    for k, (f, key, src) in F.PROFILES.items():
        p = F.J(f)[key]
        add(f"profile wall µs/layer {k} (768, bucket 128)", f"{p['wall_us']:.1f} (cc {p['cc_us']:.1f})", f"`results/phase1/{f}` [{key}] ({src})")
    # agents
    meta = F.J("t4/agent_study_meta.json")
    for run in meta["runs"]:
        rep = [p for p in sorted(os.listdir(f"{P1}/t4/reports")) if p.startswith(f"agent_{run}_")]
        add(f"agent run {run}: iteration reports", ", ".join(rep), "`results/phase1/t4/reports/`, `t4/<run>_<iter>/feedback.txt`; table 6 in TABLES.md")
    with open(a.out, "w") as f:
        f.write("# Poster numbers and their sources\n\nGenerated by `tools/make_numbers.py`. Latency: TKG p50/p99, mean ± std "
                "over separate-process runs in the final session (T6, Nov 27; 8192 from Q4b T10, Nov 26). Accuracy: frozen "
                "validator v1.1 vs the fp32 golden.\n\n| Number | Value | Source |\n|---|---|---|\n")
        for k, v, s in rows:
            f.write(f"| {k} | {v} | {s} |\n")
    print(len(rows), "numbers")


if __name__ == "__main__":
    main()
