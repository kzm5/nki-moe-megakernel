#!/usr/bin/env bash
# Q4 device chain (each step takes the device lock): T3 captures, T5 CPU emulation, Eh compiles, T4 at 2048, T2 bench.
# Usage: q4_chain.sh STEP...   (steps: t3 t5 eh t4_2048 t2_2048 t2_4096 t4_4096)
Q=/home/ubuntu/qwen-30b-a3b; V=/home/ubuntu/runs/p1/val
MK=/home/ubuntu/nki-moe-megakernel; T=$MK/tools; WT=$MK/.claude/worktrees
DEV=$WT/q1q2/tools/validator/device.py; VAL=$WT/q1q2/tools/validator/validate.py
C=$WT/q1q2/results/phase1/corpus/corpus.json
PY=~/venv-neuron-229/bin/python
E_MOD=(--moe-fused-nki-kernel-enabled --qwen megakernels.qwen3_moe.qwen_attn_e)
COMMON=(--on-device-sampling --top-k 1 --do-sample '' --output-logits)
L=~/runs/p1/q4_chain.log
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
warm() { for f in $Q/hf_model/model-*.safetensors; do dd if=$f of=/dev/null bs=16M 2>/dev/null; done; }
st() { grep -o 'EXIT_STATUS=[0-9]*' ~/runs/$1.log | tail -1; }
ev() { local s=fail; [ "$(st $1)" = EXIT_STATUS=0 ] && s=ok
  $PY ~/runs/ledger_event.py "$1_$(date -u +%Y%m%d)" ~/runs/$1.log status=$s "${@:2}" > /dev/null; }

t3() {  # three more teacher-forced captures of V at 768, each vs the first two
  for k in 3 4 5; do
    name=p1_q4_t3_V_r2r$k; out=$V/agent_c1_2_rerun$k
    [ -f $V/agent_c1_2_r2r$k.json ] && continue
    ( exec 9> ~/runs/device.lock; flock 9; warm; log "start $name"
      MK_DIR=$WT/q3/wt/c1_35309c3 MK_AGENT_VARIANT=1 ~/runs/run_py.sh $name $DEV nki $Q/agent_c1_2 $out \
        --corpus $C --refs ~/runs/p1/refs --tf )
    ev $name kind=validator_device task=Q4-T3 backend=V seq_len=768 notes="V cross-process capture $k"
    $PY $T/r2r_compare.py $V/agent_c1_2 $out --out $V/agent_c1_2_r2r$k.json >> $L 2>&1
    $PY $T/r2r_compare.py $V/agent_c1_2_rerun $out --out $V/agent_c1_2_r2r${k}_vs2.json >> $L 2>&1
    log "end $name $(st $name)"
    for j in $V/agent_c1_2_r2r$k.json $V/agent_c1_2_r2r${k}_vs2.json; do
      $PY -c "import json,sys; d=json.load(open('$j')); sys.exit(d['n_positions'] != 7264 or d['positions_differ'] or d['max_abs_logit_diff'])" \
        || { log "T3 STOP: $j not bit-identical (or missing)"; touch ~/runs/p1/q4_STOP; exit 1; }
    done
  done
}

t5() {
  ( exec 9> ~/runs/device.lock; flock 9; log "start h24"
    ~/runs/run_cpu.sh p1_q4_h24 $T/h24_allreduce.py --out $MK/results/phase1/h24.json )
  ev p1_q4_h24 kind=golden_emulation task=Q4-T5 notes="H2.4 all-reduce precision emulation"
  log "end h24 $(st p1_q4_h24)"
}

eh() {  # compile Eh at 2048 and 4096 (the compile run's benchmark is not counted)
  for s in 2048 4096; do
    d=$Q/traced_Eh_trn2_229_s$s; name=p1_q4_compile_Eh_s$s
    [ -f $d/model.pt ] && continue
    ( exec 9> ~/runs/device.lock; flock 9; warm; log "start $name"
      MK_DIR=$WT/q3/wt/t3 ATTN_E_HOIST=1 bash ~/runs/run_job.sh $name $d "${E_MOD[@]}" --seq-len $s "${COMMON[@]}" > /dev/null 2>&1 )
    ev $name kind=compile task=Q4-T2 backend=Eh seq_len=$s notes="Eh compile (a66879a, ATTN_E_HOIST=1)"
    log "end $name $(st $name)"
  done
}

t4() {  # frozen validator v1.1 at --seq-len S: XLA (compiled here) first as the logits baseline, then B, C, V
  s=$1
  for c in xla B C V; do
    case $c in
      xla) kind=xla; d=$Q/traced_xla_trn2_229_s${s}_greedy; env="MK_DIR=$WT/q1q2" ;;
      B) kind=libmoe; d=$Q/traced_libmoe_trn2_229_s${s}_greedy_sweep; env="MK_DIR=$WT/q1q2" ;;
      C) kind=nki; d=$Q/traced_nki_trn2_229_s${s}_greedy_sweep; env="MK_DIR=$WT/q1q2" ;;
      V) kind=nki; d=$Q/traced_V_q3final_s$s; env="MK_DIR=$WT/q3/wt/c1_35309c3 MK_AGENT_VARIANT=1" ;;
    esac
    name=p1_q4_t4_${c}_s$s; out=$V/${c}_s${s}_plain
    if [ ! -f $out/device.json ]; then
      ( exec 9> ~/runs/device.lock; flock 9; warm; log "start $name"
        # shellcheck disable=SC2086
        env $env ~/runs/run_py.sh $name $DEV $kind $d $out --corpus $C --refs ~/runs/p1/refs --tf --ppl --repeat --seq-len $s )
      ev $name kind=validator_device task=Q4-T4 backend=$c seq_len=$s notes="frozen device.py --seq-len $s"
      log "end $name $(st $name)"
    fi
    [ -f $out/device.json ] || { log "no capture for $name"; continue; }
    $PY $VAL score --cand $out --base-logits $V/xla_s${s}_plain --out $out/report.json > $out/score.log 2>&1
    $PY $VAL score --cand $out --out $out/report_base768.json > $out/score_base768.log 2>&1
  done
}

for step in "$@"; do
  [ -f ~/runs/p1/q4_STOP ] && { log "STOP file present; chain halted before $step"; exit 1; }
  log "STEP $step"
  case $step in
    t3) t3 || exit 1 ;; t5) t5 ;; eh) eh ;;
    t4_2048) t4 2048 ;; t4_4096) t4 4096 ;;
    t2_2048) bash $T/bench_q4.sh q4_t2 2048 3 B C V Eh ;;
    t2_4096) bash $T/bench_q4.sh q4_t2 4096 3 B C V Eh ;;
  esac
  log "STEP_DONE $step"
done
log "CHAIN_DONE $*"
