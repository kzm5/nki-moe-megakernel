#!/usr/bin/env bash
# Second teacher-forced capture of an agent build with the frozen device.py, then r2r_compare against the first.
# Usage: r2r_rerun.sh RUN ITER WORKTREE
run=$1; it=$2; wt=$3
Q=/home/ubuntu/qwen-30b-a3b; V=/home/ubuntu/runs/p1/val
DEV=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2/tools/validator/device.py
C=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2/results/phase1/corpus/corpus.json
exec 9> ~/runs/device.lock; flock 9
for f in $Q/hf_model/model-*.safetensors; do dd if=$f of=/dev/null bs=16M 2>/dev/null; done
MK_DIR=$wt MK_AGENT_VARIANT=1 ~/runs/run_py.sh p1_q3_r2r_${run}_$it $DEV nki $Q/agent_${run}_$it $V/agent_${run}_${it}_rerun \
  --corpus $C --refs /home/ubuntu/runs/p1/refs --tf
~/venv-neuron-229/bin/python /home/ubuntu/nki-moe-megakernel/.claude/worktrees/q3/tools/r2r_compare.py \
  $V/agent_${run}_$it $V/agent_${run}_${it}_rerun --out $V/agent_${run}_${it}_r2r.json
