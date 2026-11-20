#!/usr/bin/env bash
# One agent-study iteration: compile the megakernel from a worktree, run the frozen validator v1.1, benchmark.
# Usage: agent_eval.sh SETUP(A|C) RUN ITER WORKTREE      (detaches; poll ~/runs/agents/RUN/ITER/DONE)
# Prints (into ~/runs/agents/RUN/ITER/feedback.txt): A = compile status/errors, validator PASS/FAIL, TKG p50;
# C = A + the validator report rows + a per-phase profile table of one decode step.
set -u
setup=$1; run=$2; it=$3; wt=$4
Q=/home/ubuntu/qwen-30b-a3b
VAL=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2/tools/validator/validate.py
TOOLS=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q3/tools
d=~/runs/agents/$run/$it; cdir=$Q/agent_${run}_$it; name=p1_q3_t4_${run}_$it
mkdir -p $d
if [ -z "${AGENT_EVAL_CHILD:-}" ]; then
  AGENT_EVAL_CHILD=1 nohup setsid bash "$0" "$@" > $d/harness.log 2>&1 < /dev/null &
  echo "started $run/$it; poll: until [ -f $d/DONE ]; do sleep 30; done; cat $d/feedback.txt"
  exit 0
fi
fb=$d/feedback.txt; : > $fb
say() { echo "$*" | tee -a $fb; }
ts() { date -u +%FT%TZ; }
exec 9> ~/runs/device.lock; flock 9
echo "$(ts) start" > $d/times
for f in $Q/hf_model/model-*.safetensors; do dd if=$f of=/dev/null bs=16M 2>/dev/null; done
git -C "$wt" rev-parse --short HEAD > $d/commit 2>/dev/null
FLAGS=(--enable-nki --seq-len 768 --on-device-sampling --top-k 1 --do-sample '' --output-logits)
# compile + frozen validator v1.1 in one process (the validator compiles when model.pt is missing)
MK_DIR=$wt MK_AGENT_VARIANT=1 MK_LOAD_PRESHARDED=1 ~/runs/run_py.sh ${name}_val $VAL run nki $cdir ~/runs/p1/val/agent_${run}_$it > /dev/null 2>&1
echo "$(ts) validated" >> $d/times
if [ ! -f $cdir/model.pt ]; then
  say "COMPILE: FAIL"
  grep -E "Error|error:|ERROR|Traceback|assert|NCC_" ~/runs/${name}_val.log | grep -v -i "warn" | tail -40 >> $fb
  echo "--- last lines" >> $fb; tail -25 ~/runs/${name}_val.log >> $fb
  touch $d/DONE; exit 0
fi
say "COMPILE: OK ($(grep -o 'Compiling and tracing time: [0-9.]*' ~/runs/${name}_val.log | tail -1))"
v=$(grep '^VALIDATE ' ~/runs/${name}_val.log | tail -1 | cut -c10-)
if [ -z "$v" ]; then
  say "VALIDATOR: ERROR (build did not load or run)"
  grep -E "Error|Traceback|assert" ~/runs/${name}_val.log | tail -20 >> $fb
  touch $d/DONE; exit 0
fi
pass=$(echo "$v" | python3 -c "import json,sys;print('PASS' if json.load(sys.stdin)['pass'] else 'FAIL')")
say "VALIDATOR v1.1: $pass"
[ "$setup" = C ] && { echo "validator report rows:" >> $fb
  python3 -c "
import json; r=json.load(open('$HOME/runs/p1/val/agent_${run}_$it/report.json'))
for k,v in r.get('rows',{}).items(): print(' ',k, json.dumps({a:b for a,b in v.items() if not isinstance(b,list)})[:400])" >> $fb 2>&1; }
# benchmark (n=2, separate processes); pre-sharded weights/ load without the sharding RAM peak
PRESH=(); [ -d $cdir/weights ] && PRESH=(--save-sharded-checkpoint --skip-sharding)
for r in 1 2; do
  MK_DIR=$wt MK_AGENT_VARIANT=1 bash ~/runs/run_job.sh ${name}_b$r $cdir "${FLAGS[@]}" "${PRESH[@]}" --skip-compile > /dev/null 2>&1
  bash ~/runs/ledger_bench.sh ${name}_b$r 768 "agent-$run-$it" "Q3 T4 $setup $run iter $it" > /dev/null 2>&1
done
echo "$(ts) benched" >> $d/times
p50=$(python3 -c "
import json,statistics as s
v=[json.load(open('$HOME/runs/${name}_b%d/benchmark_report.json'%r))['token_generation_model']['latency_ms_p50'] for r in (1,2)]
print(f'{s.mean(v):.3f} ms (runs {v[0]:.3f}, {v[1]:.3f})')" 2>/dev/null || echo "n/a (benchmark failed)")
say "TKG p50: $p50   (unmodified megakernel: ~10.29-10.36 ms in recent sessions)"
if [ "$setup" = C ]; then
  bash ~/runs/prof/run_prof.sh ${name}_prof nki $cdir 768 > /dev/null 2>&1
  P=~/runs/prof/${name}_prof; best=""; bt=99
  for f in $P/small_bucket/*/*/*.ntff; do
    t=$(/opt/aws/neuron/bin/neuron-profile view -n $P/neffs/tkg_bucket0.neff -s $f --output-format summary-json 2>/dev/null \
        | python3 -c "import json,sys;print(list(json.load(sys.stdin).values())[0]['total_time'])")
    python3 -c "import sys;sys.exit(0 if $t<$bt else 1)" && { bt=$t; best=$f; }
  done
  /opt/aws/neuron/bin/neuron-profile view -n $P/neffs/tkg_bucket0.neff -s $best --output-format json \
    --output-file $d/prof.json --ignore-event-trace --ignore-nc-buf-usage > /dev/null 2>&1
  echo "profile (least-skewed rank, bucket 128; us per layer; NTFF/NEFF in $P, instruction JSON $d/prof.json):" >> $fb
  ~/venv-neuron-229/bin/python $TOOLS/phase_attrib.py $d/prof.json --out $d/phase.json --src-root $wt >> $fb 2>&1
  echo "$(ts) profiled" >> $d/times
fi
python3 - "$run" "$it" "$setup" "$pass" "$p50" <<'EOF' >> ~/runs/ledger.jsonl
import json, sys
run, it, setup, v, p = sys.argv[1:]
print(json.dumps({"run_id": f"p1_q3_t4_{run}_{it}", "kind": "agent_iteration", "phase": 1, "task": "Q3-T4", "setup": setup,
                  "validator": {"spec": "1.1", "verdict": v}, "perf": {"tkg_p50": p}}))
EOF
touch $d/DONE
