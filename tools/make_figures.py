"""Final tables and figures from the ledger and the validator / profile / emulation reports (no hand-typed numbers).

Computes (each table also as CSV; figures as PNG with the CSV they plot):
    1 latency vs sequence length (T6 session + T10 at 8192)       fig1_latency.{png,csv}
    2 C's gain over B at 768 split along B -> E -> Eh -> C -> V    fig2_gain_768.{png,csv}
    3 accuracy vs fp32 golden: XLA, B, C, E, Eh, V                 tab3_accuracy.csv
    4 error vs depth (H2.6) and H2.2 / H2.4 precision effects       fig4_depth.{png,csv}, tab4_precision.csv
    5 per-layer profile, Table-5 columns: B, C, E, Eh, V           tab5_profile.csv
    6 agent study A vs C per run                                   tab6_agents.csv
    7 gates 1 and 4                                                tab7_gates.csv
    8 error attribution along the swap chain XLA -> B -> E -> Eh -> C -> V   tab8_attribution.csv
    9 long context: latency 768-8192 and validation at 2048        tab9_long.csv
    TABLES.md with every table and the ledger run ids behind each latency number.

Usage:
    make_figures.py --out results/final [--t6 q5_t6] [--t10 q4b_t10]
"""
import argparse, csv, glob, json, os, re, statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = f"{ROOT}/results/phase1"
SEQS = (768, 2048, 4096, 8192)
COLORS = {"A": "#7f7f7f", "B": "#1f77b4", "C": "#d62728", "E": "#9467bd", "Eh": "#ff7f0e", "V": "#2ca02c"}
NAMES = {"A": "A (XLA)", "B": "B (lib MoE)", "C": "C (megakernel)", "E": "E (unfused attn)", "Eh": "Eh (E, hoisted setup)",
         "V": "V (C + AG collectives)", "xla": "XLA"}
ACC_REPORTS = {"xla": "reports_v11/xla_self.json", "B": "reports_v11/libmoe.json", "E": "reports_v11/E_cap.json",
               "Eh": "t3/Eh_plain_report.json", "C": "reports_v11/nki.json", "V": "t4/reports/agent_c1_2.json"}
PROFILES = {"B": ("profiles/libmoe_s768.json", "libmoe_s768_b128", "Phase 0 profile, Nov 12"),
            "C": ("profiles/Cq3_s768.json", "Cq3_s768_b128", "Q3 re-profile, Nov 19"),
            "E": ("profiles/E_s768.json", "E_s768_b128", "Q2 T9"), "Eh": ("profiles/Eh_s768.json", "Eh_s768_b128", "Q3 T3"),
            "V": ("profiles/V_s768.json", "V_s768_b128", "Q3 T4 C1 iter 2")}
sd = lambda x: st.stdev(x) if len(x) > 1 else 0.0
J = lambda p: json.load(open(f"{P1}/{p}"))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md_table(rows):
    keys = list(rows[0])
    fmt = lambda v: (f"{v:.3f}" if abs(v) >= 1 else f"{v:.4g}") if isinstance(v, float) else ("–" if v is None else str(v))
    return "\n".join(["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] +
                     ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows])


def bench(tags):
    """{(cfg, seq): {"p50": [...], "p99": [...], "ids": [...], "rounds": [...]}} from ledger rows of the given tags."""
    out = {}
    for line in open(f"{P1}/ledger.jsonl"):
        e = json.loads(line)
        m = re.match(r"p1_(.+?)_([A-Za-z]+)_s(\d+)_r(\d+)_\d+$", e["run_id"])
        if not m or m.group(1) not in tags or e.get("status") != "ok" or not e.get("tkg_latency_ms_p50"):
            continue
        k = (m.group(2), int(m.group(3)))
        d = out.setdefault(k, {"p50": [], "p99": [], "ids": [], "rounds": []})
        d["p50"].append(e["tkg_latency_ms_p50"]); d["p99"].append(e["tkg_latency_ms_p99"])
        d["ids"].append(e["run_id"]); d["rounds"].append(int(m.group(4)))
    return out


def stat(b, c, s):
    d = b.get((c, s))
    return None if not d else {"n": len(d["p50"]), "p50": st.mean(d["p50"]), "p50_sd": sd(d["p50"]),
                                "p99": st.mean(d["p99"]), "p99_sd": sd(d["p99"]), "ids": d["ids"]}


def delta(b, x, y, s):
    a, c = stat(b, x, s), stat(b, y, s)
    if not a or not c:
        return None
    return {"ms": a["p50"] - c["p50"], "two_sigma": 2 * ((a["p50_sd"] ** 2 + c["p50_sd"] ** 2) / 2) ** 0.5}


def acc_row(k):
    r = J(ACC_REPORTS[k])
    x = r["rows"]
    ob = x["op_boundary"].get("per_boundary", {})
    return {"config": NAMES.get(k, k), "v1.1": "PASS" if r["pass"] else "FAIL",
            "rho_attn": ob.get("attn", {}).get("rho_median"), "rho_moe": ob.get("moe", {}).get("rho_median"),
            "rho_resid": ob.get("resid", {}).get("rho_median"), "router_flip_pct": x["router"].get("flip_pct_cand"),
            "kl_mean": x["logits_tf"]["kl_mean_cand"], "kl_ratio_vs_xla": x["logits_tf"]["kl_ratio"],
            "top1_golden": x["logits_tf"]["top1_golden_cand"], "ppl": x["perplexity"]["ppl_cand"],
            "gross": x["worst_pos"]["gross_count"], "run_to_run": x["run_to_run"]["status"], "source": ACC_REPORTS[k]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--t6", default="q5_t6")
    ap.add_argument("--t10", default="q4b_t10")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    o = lambda f: os.path.join(a.out, f)
    md = ["# Final tables and figures\n", "Generated by `tools/make_figures.py` from `results/phase1/ledger.jsonl` and the "
          "reports under `results/phase1/`; TKG p50/p99 in ms, mean ± std over separate-process runs.\n"]
    b = bench({a.t6, a.t10})

    # 1 latency vs seq
    rows = []
    for c in ("A", "B", "C", "E", "Eh", "V"):
        for s in SEQS:
            x = stat(b, c, s)
            if x:
                rows.append({"config": c, "seq": s, "n": x["n"], "p50_mean": x["p50"], "p50_std": x["p50_sd"],
                             "p99_mean": x["p99"], "p99_std": x["p99_sd"], "ledger_ids": " ".join(x["ids"])})
    if rows:
        write_csv(o("fig1_latency.csv"), rows)
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in ("B", "C", "V", "Eh", "A"):
            pts = [r for r in rows if r["config"] == c]
            if pts:
                ax.errorbar([r["seq"] for r in pts], [r["p50_mean"] for r in pts], yerr=[r["p50_std"] for r in pts],
                            marker="o", capsize=3, label=NAMES[c], color=COLORS[c])
        ax.set_xscale("log", base=2); ax.set_xticks(SEQS); ax.set_xticklabels(SEQS)
        ax.set_xlabel("compiled sequence length"); ax.set_ylabel("TKG p50 (ms/step)"); ax.legend(fontsize=8)
        ax.set_title("Decode latency vs sequence length (trn2, TP=4, bs=1)")
        fig.tight_layout(); fig.savefig(o("fig1_latency.png"), dpi=150); plt.close(fig)
        md += ["## 1. Latency vs sequence length\n", md_table([{k: v for k, v in r.items() if k != "ledger_ids"} for r in rows]),
               "\nLedger ids: `fig1_latency.csv`.\n"]

    # 2 gain decomposition at 768
    chain = [("B", "E", "unfused attention kernel (+ call cost)"), ("E", "Eh", "per-call setup hoisted"),
             ("Eh", "C", "fusion (call boundary, MoE kernel)"), ("C", "V", "collectives (AG + local sum)")]
    rows = []
    for x, y, what in chain:
        d = delta(b, y, x, 768)
        if d:
            rows.append({"step": f"{x} -> {y}", "what": what, "delta_ms": d["ms"], "two_pooled_std": d["two_sigma"]})
    if rows and stat(b, "B", 768) and stat(b, "C", 768):
        bb, cc = stat(b, "B", 768)["p50"], stat(b, "C", 768)["p50"]
        rows.append({"step": "B -> C (total)", "what": "C's gain over B", "delta_ms": cc - bb,
                     "two_pooled_std": delta(b, "C", "B", 768)["two_sigma"]})
        write_csv(o("fig2_gain_768.csv"), rows)
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        level = bb
        for i, r in enumerate(rows[:-1]):
            ax.bar(i, r["delta_ms"], bottom=level, color="#2ca02c" if r["delta_ms"] < 0 else "#d62728",
                   yerr=r["two_pooled_std"] / 2, capsize=3)
            level += r["delta_ms"]
        ax.axhline(bb, color=COLORS["B"], ls="--", lw=1, label=f"B {bb:.3f}")
        ax.axhline(cc, color=COLORS["C"], ls=":", lw=1, label=f"C {cc:.3f}")
        ax.set_xticks(range(len(rows) - 1)); ax.set_xticklabels([r["step"] for r in rows[:-1]], fontsize=8)
        ax.set_ylabel("TKG p50 (ms)"); ax.legend(fontsize=8); ax.set_title("Seq 768: B -> E -> Eh -> C -> V (T6 session)")
        fig.tight_layout(); fig.savefig(o("fig2_gain_768.png"), dpi=150); plt.close(fig)
        md += ["## 2. Where C's gain over B comes from (768)\n", md_table(rows), ""]

    # 3 accuracy
    rows = [acc_row(k) for k in ("xla", "B", "E", "Eh", "C", "V")]
    write_csv(o("tab3_accuracy.csv"), rows)
    md += ["## 3. Accuracy vs fp32 golden (frozen validator v1.1, 7,264 positions)\n",
           md_table([{k: v for k, v in r.items() if k != "source"} for r in rows]),
           "\nρ and router flips need a capture build (none for C, V; Eh is bit-identical to E). Sources in the CSV.\n"]

    # 4 error vs depth + precision
    eb, h22, h24 = J("error_budget/xla_B_E.json"), J("h22.json"), J("h24.json")
    rows = []
    for L in range(48):
        r = {"layer": L}
        for k in ("xla", "B", "E"):
            r[f"{k}_resid_rel_l2"] = eb[k]["resid"]["rel_l2"][L]
        for v in h22["results"]:
            r[f"h22_{v['variant']}"] = v["resid_rel_l2"][L]
        for v in h24["results"]:
            r[f"h24_{v['variant']}"] = v["resid_rel_l2"][L]
        rows.append(r)
    write_csv(o("fig4_depth.csv"), rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, c in (("xla", "A"), ("B", "B"), ("E", "E")):
        a1.plot([r["layer"] for r in rows], [r[f"{k}_resid_rel_l2"] for r in rows], label=NAMES[k if k != "xla" else "A"], color=COLORS[c])
    a1.set_xlabel("layer"); a1.set_ylabel("residual rel-L2 vs fp32 golden"); a1.set_title("Device error vs depth (H2.6)"); a1.legend(fontsize=8)
    for v in h22["results"] + h24["results"]:
        src = "H2.2" if v in h22["results"] else "H2.4"
        if v["variant"] != "fp32":
            a2.plot(range(48), v["resid_rel_l2"], label=f"{src} {v['variant']}")
    a2.set_xlabel("layer"); a2.set_title("CPU emulation: one precision change each"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(o("fig4_depth.png"), dpi=150); plt.close(fig)
    prec = [{"study": "H2.2 expert GEMMs", "variant": v["variant"], "kl_mean": v["kl_mean"], "kl_max": v["kl_max"],
             "top1_golden": v["top1_golden"], "resid_L47": v["resid_rel_l2"][47]} for v in h22["results"]]
    prec += [{"study": "H2.4 TP all-reduce", "variant": v["variant"], "kl_mean": v["kl_mean"], "kl_max": v["kl_max"],
              "top1_golden": v["top1_golden"], "resid_L47": v["resid_rel_l2"][47]} for v in h24["results"]]
    write_csv(o("tab4_precision.csv"), prec)
    md += ["## 4. Error vs depth and precision effects\n", "![depth](fig4_depth.png)\n", md_table(prec), ""]

    # 5 profile
    rows = []
    for k, (f, key, src) in PROFILES.items():
        p = J(f)[key]
        rows.append({"config": k, "wall_us": p["wall_us"], "sync_us": p["sync_us"], "dma_active_us": p["dma_active_us"],
                     "tensor_us": p["tensor_us"], "vector_us": p["vector_us"], "scalar_us": p["scalar_us"],
                     "gpsimd_us": p["gpsimd_us"], "cc_us": p["cc_us"], "dma_packets": p["dma_packets_per_layer"],
                     "dma_avg_bytes": p["dma_avg_packet_bytes"], "hbm_read_mb": p["hbm_read_mb_per_layer"],
                     "spill_bytes": p["spill_reload_bytes"] + p["spill_save_bytes"], "source": src})
    write_csv(o("tab5_profile.csv"), rows)
    md += ["## 5. Per-layer decode profile (µs/layer, bucket 128, seq 768, least-skewed rank)\n",
           md_table(rows), "\nB's row is the Phase 0 profile (same SDK and method, earlier day).\n"]

    # 6 agent study
    meta = J("t4/agent_study_meta.json")
    base = meta["base_harness_p50_ms"]["value"]
    rows = []
    for run, m in meta["runs"].items():
        its = sorted((p for p in glob.glob(f"{P1}/t4/{run}_*") if re.fullmatch(rf"{run}_\d+", os.path.basename(p))
                      and os.path.isdir(p)), key=lambda p: int(p.rsplit("_", 1)[1]))
        n_it = len(its)
        compiled = elig = nondet = passed = 0
        best = None
        for d in its:
            fb = open(f"{d}/feedback.txt").read() if os.path.exists(f"{d}/feedback.txt") else ""
            if "COMPILE: OK" not in fb:
                continue
            compiled += 1
            rp = f"{P1}/t4/reports/agent_{os.path.basename(d)}.json"
            if not os.path.exists(rp):
                continue
            r = J(f"t4/reports/agent_{os.path.basename(d)}.json")
            det = r["rows"]["run_to_run"]["status"] == "bit_identical"
            passed += r["pass"]; nondet += not det
            mm = re.search(r"TKG p50: ([\d.]+) ms", fb)
            if r["pass"] and det and mm:
                elig += 1
                best = min(best or 1e9, float(mm.group(1)))
        rows.append({"run": run, "setup": m["setup"], "iterations": n_it, "compiled": compiled,
                     "compile_failures": n_it - compiled, "pass_v1.1": passed, "nondeterministic": nondet,
                     "eligible": elig, "best_eligible_ms": best,
                     "best_vs_base_ms": None if best is None else best - base,
                     "wall_min": m["wall_min"], "idle_min": m["idle_min"], "active_min": m["wall_min"] - m["idle_min"],
                     "interventions": m["interventions"], "guard_blocks": m["guard_blocks"],
                     "tooling_failures": "; ".join(m["tooling_failures"]) or "none"})
    write_csv(o("tab6_agents.csv"), rows)
    md += ["## 6. Agent study (setup A = verdict + p50; setup C = + validator rows + profile)\n", md_table(rows),
           f"\nHarness latencies (n=2); base (unmodified) {base} ms. Tooling failures are listed separately from setup "
           "effects; wall/idle/interventions transcribed in `t4/agent_study_meta.json`.\n"]

    # 7 gates
    g1 = {m: J(f"reports_v11/{m}.json")["pass"] for m in ("m1", "m2", "m3", "m4")}
    rows = [{"gate": "Gate 1 (validator catches the planted faults, benign control passes)",
             "numbers": f"m1 {'pass' if g1['m1'] else 'caught'}, m2 {'pass' if g1['m2'] else 'caught'}, "
                        f"m4 {'pass' if g1['m4'] else 'caught'}, m3 (control) {'pass' if g1['m3'] else 'FAIL'}",
             "verdict": "met" if (not g1["m1"] and not g1["m2"] and not g1["m4"] and g1["m3"]) else "not met"}]
    for s in (768, 2048, 4096, 8192):
        x, y = stat(b, "V", s), stat(b, "B", s)
        if x and y:
            gain = 100 * (y["p50"] - x["p50"]) / y["p50"]
            rows.append({"gate": f"Gate 4 (best validated variant V >= 5% faster than B) at {s}" +
                                 (" — judged point" if s == 768 else " — secondary"),
                         "numbers": f"V {x['p50']:.3f} vs B {y['p50']:.3f} ms: {gain:.2f}%",
                         "verdict": ("met" if gain >= 5 else "not met") if s == 768 else (f"{'≥' if gain >= 5 else '<'} 5%")})
    write_csv(o("tab7_gates.csv"), rows)
    md += ["## 7. Gates\n", md_table(rows), ""]

    # 8 error attribution along the swap chain
    chain = [("xla", "A", "start"), ("B", "B", "MoE kernel"), ("E", "E", "attention kernel (unfused)"),
             ("Eh", "Eh", "per-call setup"), ("C", "C", "fusion"), ("V", "V", "collectives")]
    rows, prev = [], None
    for k, c, what in chain:
        r = acc_row(k)
        lat = stat(b, c, 768)
        row = {"step": NAMES.get(k, k), "swap": what, "rho_resid": r["rho_resid"], "router_flip_pct": r["router_flip_pct"],
               "kl_ratio": r["kl_ratio_vs_xla"], "ppl": r["ppl"], "p50_768": lat["p50"] if lat else None}
        if prev:
            row["d_kl_ratio"] = row["kl_ratio"] - prev["kl_ratio"]
            row["d_ppl"] = row["ppl"] - prev["ppl"]
            row["d_p50"] = None if row["p50_768"] is None or prev["p50_768"] is None else row["p50_768"] - prev["p50_768"]
        else:
            row.update(d_kl_ratio=None, d_ppl=None, d_p50=None)
        rows.append(row); prev = row
    write_csv(o("tab8_attribution.csv"), rows)
    md += ["## 8. Error attribution along the swap chain (accuracy: v1.1 reports; latency: T6 at 768)\n", md_table(rows), ""]

    # 9 long context
    rows = []
    for c in ("B", "C", "V", "Eh"):
        for s in SEQS:
            x = stat(b, c, s)
            if x:
                rows.append({"kind": "latency", "config": c, "seq": s, "value": x["p50"], "std": x["p50_sd"], "n": x["n"],
                             "note": "T10 (final 8192 numbers)" if s == 8192 else "T6"})
    for c in ("xla", "B", "C", "V"):
        for sub, f, note in (("v1 corpus (positions < 768)", f"t4_s2048/{c}.json", "frozen v1.1"),
                             ("long corpus (positions 1,400-1,958)", f"corpus_long/reports/{c}_s2048.json", "supplementary")):
            if os.path.exists(f"{P1}/{f}"):
                r = J(f)
                r = r.get("report", r)
                rows.append({"kind": f"validation 2048, {sub}", "config": c, "seq": 2048,
                             "value": r["rows"]["logits_tf"]["kl_ratio"], "std": None, "n": r["rows"]["logits_tf"]["n_positions"],
                             "note": f"{note}: {'PASS' if r['pass'] else 'FAIL'}, KL ratio vs XLA"})
    if rows:
        write_csv(o("tab9_long.csv"), rows)
        md += ["## 9. Long context (latency 768–8192; validation at 2048, long corpus labelled supplementary)\n", md_table(rows), ""]
    open(o("TABLES.md"), "w").write("\n".join(md) + "\n")
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
