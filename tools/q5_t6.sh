#!/usr/bin/env bash
# Q5 T6: final interleaved session, one order per round (all poster timing numbers).
T=/home/ubuntu/nki-moe-megakernel/tools
bash $T/bench_q4.sh q5_t6 768 5 A B C E Eh V
bash $T/bench_q4.sh q5_t6 2048 3 B C V Eh
bash $T/bench_q4.sh q5_t6 4096 3 B C V Eh
echo "$(date -u +%FT%TZ) T6_DONE" >> ~/runs/p1/bench_q5_t6.log
