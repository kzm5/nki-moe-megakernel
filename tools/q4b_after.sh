#!/usr/bin/env bash
# After Q4b T10: T11 validate proof (clean worktree), T4 at 4096 (fits before Fri 12:00 ET), then Q5 T6.
MK=/home/ubuntu/nki-moe-megakernel; T=$MK/tools; L=~/runs/p1/q4b_after.log
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
# wait for the T10 bench (it holds the device lock for its whole loop)
while pgrep -f "bench_q4.sh q4b_t10" > /dev/null; do sleep 30; done
log "start T11 validate proof"
(cd $MK/.claude/worktrees/q4b-repro2 && bash scripts/reproduce/reproduce.sh validate ~/runs/repro_proof_config.sh) >> $L 2>&1
log "end T11 validate proof"
log "start T4 4096"
bash $T/q4_chain.sh t4_4096
log "end T4 4096"
log "start T6"
bash $T/q5_t6.sh
log "end T6"
