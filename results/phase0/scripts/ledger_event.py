"""Append a non-benchmark run to ~/runs/ledger.jsonl.
Usage: ledger_event.py RUN_ID LOG KEY=JSON_OR_STR ...   (wall/peak RSS parsed from /usr/bin/time -v in LOG)"""
import json, os, re, subprocess, sys
run_id, log = sys.argv[1:3]
e = {"run_id": run_id, "venv": os.path.expanduser("~/venv-neuron-229")}
start = os.path.expanduser(f"~/runs/{os.path.basename(log)[:-4]}.start")
if os.path.exists(start):
    e["timestamp"] = open(start).read().strip()
e["git_sha"] = subprocess.run(["git", "-C", os.path.expanduser("~/nki-moe-megakernel/.claude/worktrees/session-rules"),
                               "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if os.path.exists(log):
    t = open(log, errors="replace").read()
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", t)
    if m:
        e["wall_seconds"] = round(sum(float(v) * 60 ** i for i, v in enumerate(reversed(m.group(1).split(":")))), 2)
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    if m:
        e["peak_rss_gib"] = round(int(m.group(1)) / 2**20, 2)
    e["log"] = log
for kv in sys.argv[3:]:
    k, v = kv.split("=", 1)
    try:
        e[k] = json.loads(v)
    except json.JSONDecodeError:
        e[k] = v
with open(os.path.expanduser("~/runs/ledger.jsonl"), "a") as f:
    f.write(json.dumps(e) + "\n")
print(json.dumps(e))
