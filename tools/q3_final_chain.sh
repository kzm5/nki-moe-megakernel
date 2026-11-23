#!/usr/bin/env bash
T=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q3/tools
export V_WT=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q3/wt/c1_35309c3 V_768=/home/ubuntu/qwen-30b-a3b/agent_c1_2
L=~/runs/p1/bench_q3_final.log
echo "$(date -u +%FT%TZ) chain start r2r c1_2" >> $L
bash $T/r2r_rerun.sh c1 2 $V_WT >> $L 2>&1
echo "$(date -u +%FT%TZ) r2r done" >> $L
bash $T/bench_q3_final.sh compile 2048 4096
bash $T/bench_q3_final.sh rounds 768 5
bash $T/bench_q3_final.sh rounds 2048 3
bash $T/bench_q3_final.sh rounds 4096 3
echo "$(date -u +%FT%TZ) CHAIN_DONE" >> $L
