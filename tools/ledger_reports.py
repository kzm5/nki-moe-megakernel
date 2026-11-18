"""Append one ledger entry per validator report: ledger_reports.py TASK NAME..."""
import json, os, subprocess, sys, time

R = os.path.expanduser(os.environ.get("REPORTS", "~/runs/p1/reports"))
WT = os.path.expanduser("~/nki-moe-megakernel/.claude/worktrees/q1q2")
task = sys.argv[1]
sha = subprocess.run(["git", "-C", WT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
with open(os.path.expanduser("~/runs/ledger.jsonl"), "a") as f:
    for b in sys.argv[2:]:
        r = json.load(open(f"{R}/{b}.json"))
        rows = r["rows"]
        ob = rows["op_boundary"].get("per_boundary", {})
        e = {"run_id": f"p1_validate{os.environ.get('TAG', '')}_{b}_{time.strftime('%Y%m%d', time.gmtime())}", "timestamp": r["scored_utc"],
             "git_sha": sha, "kind": "validator", "phase": 1, "task": task, "setup": b,
             "config": r["cand_device"], "report": f"{R}/{b}.json",
             "validator": {"pass": r["pass"], "rows_failed": r["rows_failed"], "rows_not_measurable": r["rows_not_measurable"],
                           "rho_median": {k: v["rho_median"] for k, v in ob.items()},
                           "rho_max": {k: v["rho_max"] for k, v in ob.items()},
                           "router_delta_pp": rows["router"].get("delta_pp"), "kl_ratio": rows["logits_tf"]["kl_ratio"],
                           "gross": rows["worst_pos"]["gross_count"], "max_kl": rows["worst_pos"]["max_kl"],
                           "ppl_rel": rows["perplexity"]["rel_delta"], "run_to_run": rows["run_to_run"]["status"],
                           "frozen_ok": r["frozen_ok"], "spec_sha": r["hashes"]["spec"],
                           "validator_sha": r["hashes"]["validator"], "corpus_sha": r["hashes"]["corpus"],
                           "golden_manifest": r["hashes"]["golden_manifest"]},
             "outcome": "pass" if r["pass"] else "fail"}
        f.write(json.dumps(e) + "\n")
        print(e["run_id"], e["outcome"], r["rows_failed"])
