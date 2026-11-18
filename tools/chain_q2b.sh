#!/usr/bin/env bash
# After T8 4096: rescore under spec 1.0 and 1.1, T9 profile of E, T8b compiles and rounds. One job at a time.
WT=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2
V=/home/ubuntu/runs/p1/val
P=~/venv-neuron-229/bin/python
L=~/runs/p1/chain_q2b.log
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
cd $WT
for step in "$@"; do
  df -h / | tail -1 >> $L; log "start $step"
  case $step in
    rescore)
      for ver in 1.0 1.1; do
        out=~/runs/p1/reports_v$(echo $ver | tr -d .)
        mkdir -p $out
        $P tools/validator/validate.py score --cand $V/xla_self --prev $V/xla_base --spec $ver --out $out/xla_self.json >> $L
        for b in m1 m2 m3 m4 libmoe nki E_plain E_cap; do
          $P tools/validator/validate.py score --cand $V/$b --spec $ver --out $out/$b.json >> $L
        done
      done ;;
    prof_E) bash ~/runs/prof/run_prof.sh p1_prof_E_s768 E /home/ubuntu/qwen-30b-a3b/traced_E_trn2_229_s768 768 ;;
    t8b_warmup) bash tools/bench_t8b.sh warmup ;;
    t8b_rounds) bash tools/bench_t8b.sh rounds 3 ;;
    *) log "unknown $step" ;;
  esac
  log "end $step"
done
log "CHAIN_DONE $*"
