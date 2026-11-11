#!/usr/bin/env bash
# Usage: ledger_bench.sh NAME SEQ BACKEND NOTES   (benchmark run via run_job.sh)
name=$1; seq=$2; backend=$3; notes=$4
. ~/venv-neuron-229/activate-neuron.sh
cd ~/runs
python ledger_add.py ${name}_20261111 ~/runs/$name.log ~/runs/$name/benchmark_report.json \
  "$(git -C ~/nki-moe-megakernel rev-parse --short HEAD)" "$(cat ~/runs/$name.start)" 0 $seq ~/venv-neuron-229 > /dev/null
python - "$backend" "$notes" <<'EOF'
import json, os, sys
p = os.path.expanduser("~/runs/ledger.jsonl"); L = open(p).read().splitlines(); d = json.loads(L[-1])
d["config"]["backend"] = sys.argv[1]; d["validated"] = False; d["notes"] = sys.argv[2]
L[-1] = json.dumps(d); open(p, "w").write("\n".join(L) + "\n")
print(d["run_id"], round(d["tkg_latency_ms_p50"], 3), round(d["tkg_latency_ms_p99"], 3), d["wall_seconds"])
EOF
