"""Print a one-block summary of validator reports: summarize_reports.py NAME..."""
import json, os, sys

R = os.path.expanduser("~/runs/p1/reports")
for b in sys.argv[1:]:
    r = json.load(open(f"{R}/{b}.json"))
    rows = r["rows"]
    print(b, json.dumps({k: r[k] for k in ("pass", "rows_failed", "rows_not_measurable", "frozen_ok")}))
    o = rows["op_boundary"].get("per_boundary", {})
    print("  rho med/max(layer)", {k: (round(v["rho_median"], 3), round(v["rho_max"], 3), v["rho_max_layer"]) for k, v in o.items()})
    ro = rows["router"]
    if ro.get("pass") is not None:
        print("  router flip pct %.3f base %.3f delta %.3f pp, max layer delta %.2f (L%d)" % (
            ro["flip_pct_cand"], ro["flip_pct_base"], ro["delta_pp"], ro["delta_pp_max_layer"], ro["max_layer"]))
    lt, w, p = rows["logits_tf"], rows["worst_pos"], rows["perplexity"]
    print("  KL %.3e (base %.3e) ratio %.3f top1 %.4f | max KL %.3f at %s (base %.3f) gross %d %s | ppl %.4f (base %.4f) rel %+.4f | r2r %s" % (
        lt["kl_mean_cand"], lt["kl_mean_base"], lt["kl_ratio"], lt["top1_golden_cand"], w["max_kl"], w["max_kl_at"],
        w["base_kl_there"], w["gross_count"], w["gross_at"][:5], p["ppl_cand"], p["ppl_base"], p["rel_delta"],
        rows["run_to_run"]["status"]))
