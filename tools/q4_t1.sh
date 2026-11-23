#!/usr/bin/env bash
# Q4 T1: compile V + trace-time asserts at 768 and run the frozen validator v1.1 on it (one process), as agent_eval.sh did.
# Usage: q4_t1.sh WORKTREE
wt=$1
Q=/home/ubuntu/qwen-30b-a3b
VAL=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2/tools/validator/validate.py
name=p1_q4_t1_V768
exec 9> ~/runs/device.lock; flock 9
for f in $Q/hf_model/model-*.safetensors; do dd if=$f of=/dev/null bs=16M 2>/dev/null; done
MK_DIR=$wt MK_AGENT_VARIANT=1 ~/runs/run_py.sh $name $VAL run nki $Q/traced_V_t1_s768 ~/runs/p1/val/V_t1_s768
python3 ~/runs/ledger_event.py ${name}_$(date -u +%Y%m%d) ~/runs/$name.log \
  backend=V seq_len=768 notes="Q4 T1: V 35309c3 + asserts ($(git -C $wt rev-parse --short HEAD)), compile + frozen validator v1.1"
touch ~/runs/$name.DONE
