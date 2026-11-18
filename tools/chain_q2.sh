#!/usr/bin/env bash
# Q2 chain, one job at a time: E capture build + validator stats, E 2048/4096 compiles, interleaved B/C/E benchmarks.
WT=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2
C=$WT/results/phase1/corpus/corpus.json
V=/home/ubuntu/runs/p1/val
L=~/runs/p1/chain_q2.log
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
cd $WT
for step in "$@"; do
  df -h / | tail -1 >> $L
  log "start $step"
  case $step in
    E_cap)
      [ -f $V/E_cap/device.json ] || ~/runs/run_py.sh p1_E_cap tools/validator/device.py E /home/ubuntu/qwen-30b-a3b/traced_E_trn2_229_s768_vcap \
        $V/E_cap --corpus $C --refs /home/ubuntu/runs/p1/refs --tf --ppl --repeat --capture ;;
    E_stats)
      [ -f $V/E_cap/stats.pt ] || ~/runs/run_py.sh p1_golden_stats_E tools/validator/golden.py --corpus $C --refs /home/ubuntu/runs/p1/refs \
        --out $V/golden_tmp --stats E_cap=$V/E_cap ;;
    warmup) bash tools/bench_bce.sh warmup ;;
    b768) bash tools/bench_bce.sh rounds 768 5 ;;
    b2048) bash tools/bench_bce.sh rounds 2048 3 ;;
    b4096) bash tools/bench_bce.sh rounds 4096 3 ;;
    *) log "unknown $step" ;;
  esac
  log "end $step"
done
log "CHAIN_DONE $*"
