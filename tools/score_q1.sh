#!/usr/bin/env bash
# Score Q1 builds with the frozen validator; reports in ~/runs/p1/reports/<build>.json
cd /home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2
V=/home/ubuntu/runs/p1/val
R=/home/ubuntu/runs/p1/reports
mkdir -p $R
P=~/venv-neuron-229/bin/python
S=tools/validator/validate.py
$P $S score --cand $V/xla_self --prev $V/xla_base --out $R/xla_self.json
for b in "$@"; do
  $P $S score --cand $V/$b --out $R/$b.json
done
