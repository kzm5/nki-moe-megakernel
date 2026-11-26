#!/usr/bin/env bash
# Q4b T10 bench after C's runtime hang at 8192: round 1 B -> V (B doubles as a device-health check), rounds 2-3 for
# the configs that ran, then one final C retry (last, so a second hang cannot contaminate a measured run).
MK=/home/ubuntu/nki-moe-megakernel; T=$MK/tools; L=~/runs/p1/q4b_t10.log
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
ok() { grep -q 'EXIT_STATUS=0' ~/runs/p1_q4b_t10_$1_s8192_r$2.log 2>/dev/null && [ -f ~/runs/p1_q4b_t10_$1_s8192_r$2/benchmark_report.json ]; }
bash $T/bench_q4.sh q4b_t10 8192 1 B V
cfgs=(); for c in B V; do ok $c 1 && cfgs+=($c); done
log "round 1 ran: ${cfgs[*]}"
[ ${#cfgs[@]} -gt 0 ] && bash $T/bench_q4.sh q4b_t10 8192 3 "${cfgs[@]}"
bash $T/bench_q4.sh q4b_t10 8192 1 C
ok C 1 && log "C retry ran" || log "C retry failed: $(grep -m1 -E 'ERROR' ~/runs/p1_q4b_t10_C_s8192_r1.log | cut -c1-200)"
log "T10_BENCH_DONE"
