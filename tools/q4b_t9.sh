#!/usr/bin/env bash
# Q4b T9: supplementary long-context validation at --seq-len 2048 with the frozen device.py / golden.py and validate_long.py.
# Usage: q4b_t9.sh STEP...   (steps: xla golden cands freeze score)
Q=/home/ubuntu/qwen-30b-a3b; P1=/home/ubuntu/runs/p1; OUT=$P1/val_long
MK=/home/ubuntu/nki-moe-megakernel; T=$MK/tools; WT=$MK/.claude/worktrees; FZ=$WT/q1q2/tools/validator
CL=$MK/results/phase1/corpus_long/corpus.json; REFS=$P1/refs_long2048; GOLD=$P1/golden_long2048
PY=~/venv-neuron-229/bin/python
L=$P1/q4b_t9.log
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
warm() { for f in $Q/hf_model/model-*.safetensors; do dd if=$f of=/dev/null bs=16M 2>/dev/null; done; }
st() { grep -o 'EXIT_STATUS=[0-9]*' ~/runs/$1.log | tail -1; }
ev() { local s=fail; [ "$(st $1)" = EXIT_STATUS=0 ] && s=ok
  $PY ~/runs/ledger_event.py "$1_$(date -u +%Y%m%d)" ~/runs/$1.log status=$s "${@:2}" > /dev/null; }
mkdir -p $OUT
exec 9> ~/runs/device.lock; flock 9

xla() {
  name=p1_q4b_t9_xla_s2048
  [ -f $OUT/xla_s2048/device.json ] && return
  warm; log "start $name"
  MK_DIR=$WT/q1q2 ~/runs/run_py.sh $name $FZ/device.py xla $Q/traced_xla_trn2_229_s2048_greedy $OUT/xla_s2048 \
    --corpus $CL --refs $OUT/xla_s2048/free --free all --tf --ppl --repeat --seq-len 2048
  ev $name kind=validator_device task=Q4b-T9 backend=xla seq_len=2048 notes="long corpus: XLA references (free) + tf + ppl"
  log "end $name $(st $name)"
  mkdir -p $REFS && cp $OUT/xla_s2048/free/expected_p*.pt $REFS/
}

golden() {
  name=p1_q4b_t9_golden
  [ -f $GOLD/layers.pt ] && return
  log "start $name"
  ~/runs/run_cpu.sh $name $FZ/golden.py --corpus $CL --refs $REFS --out $GOLD --write-golden
  ev $name kind=golden task=Q4b-T9 notes="fp32 golden on the long corpus"
  log "end $name $(st $name)"
}

cands() {
  for c in B C V; do
    case $c in
      B) kind=libmoe; d=$Q/traced_libmoe_trn2_229_s2048_greedy_sweep; env="MK_DIR=$WT/q1q2" ;;
      C) kind=nki; d=$Q/traced_nki_trn2_229_s2048_greedy_sweep; env="MK_DIR=$WT/q1q2" ;;
      V) kind=nki; d=$Q/traced_V_q3final_s2048; env="MK_DIR=$WT/q3/wt/c1_35309c3 MK_AGENT_VARIANT=1" ;;
    esac
    name=p1_q4b_t9_${c}_s2048
    [ -f $OUT/${c}_s2048/device.json ] && continue
    warm; log "start $name"
    # shellcheck disable=SC2086
    env $env ~/runs/run_py.sh $name $FZ/device.py $kind $d $OUT/${c}_s2048 --corpus $CL --refs $REFS --tf --ppl --repeat --seq-len 2048
    ev $name kind=validator_device task=Q4b-T9 backend=$c seq_len=2048 notes="long corpus tf + ppl"
    log "end $name $(st $name)"
  done
}

freeze() { [ -f $MK/results/phase1/frozen_long.json ] || $PY $T/validate_long.py --freeze >> $L 2>&1; }

score() {
  for c in xla B C V; do
    $PY $T/validate_long.py --cand $OUT/${c}_s2048 --base-logits $OUT/xla_s2048 --out $OUT/${c}_s2048/report_long.json \
      > $OUT/${c}_s2048/score_long.log 2>&1
    log "score $c $(grep VALIDATE_LONG $OUT/${c}_s2048/score_long.log)"
  done
}

for s in "$@"; do log "STEP $s"; $s; log "STEP_DONE $s"; done
log "T9_DONE $*"
