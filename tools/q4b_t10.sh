#!/usr/bin/env bash
# Q4b T10: compile B, C, V at 8192 (a failed compile is recorded and skipped), then interleaved bench B -> C -> V, n=3.
# Usage: q4b_t10.sh [CONFIGS...]   (default: B C V)
Q=/home/ubuntu/qwen-30b-a3b; MK=/home/ubuntu/nki-moe-megakernel; T=$MK/tools; WT=$MK/.claude/worktrees
S=8192; PY=~/venv-neuron-229/bin/python
COMMON=(--on-device-sampling --top-k 1 --do-sample '' --output-logits)
E_MOD=(--moe-fused-nki-kernel-enabled --qwen megakernels.qwen3_moe.qwen_attn_e)
L=~/runs/p1/q4b_t10.log
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
warm() { for f in $Q/hf_model/model-*.safetensors; do dd if=$f of=/dev/null bs=16M 2>/dev/null; done; }
st() { grep -o 'EXIT_STATUS=[0-9]*' ~/runs/$1.log | tail -1; }
cfgs=("${@:-B C V}"); [ $# -eq 0 ] && cfgs=(B C V)
for c in "${cfgs[@]}"; do
  case $c in
    B) d=$Q/traced_libmoe_trn2_229_s${S}_greedy_sweep; env="MK_DIR=$WT/q1q2"; flags=(--moe-fused-nki-kernel-enabled) ;;
    C) d=$Q/traced_nki_trn2_229_s${S}_greedy_sweep; env="MK_DIR=$WT/q1q2"; flags=(--enable-nki) ;;
    V) d=$Q/traced_V_q3final_s$S; env="MK_DIR=$WT/q3/wt/c1_35309c3 MK_AGENT_VARIANT=1"; flags=(--enable-nki) ;;
    Eh) d=$Q/traced_Eh_trn2_229_s$S; env="MK_DIR=$WT/q3/wt/t3 ATTN_E_HOIST=1"; flags=("${E_MOD[@]}") ;;
  esac
  name=p1_q4b_t10_compile_${c}_s$S
  [ -f $d/model.pt ] && continue
  ( exec 9> ~/runs/device.lock; flock 9; warm; df -h / | tail -1 >> $L; log "start $name"
    # shellcheck disable=SC2086
    env $env bash ~/runs/run_job.sh $name $d "${flags[@]}" --seq-len $S "${COMMON[@]}" > /dev/null 2>&1 )
  s=fail; [ "$(st $name)" = EXIT_STATUS=0 ] && [ -f $d/model.pt ] && s=ok
  err=""; [ $s = fail ] && err=$(grep -m1 -E "Error|error:|ERROR|Traceback|assert|NCC_" ~/runs/$name.log | cut -c1-300)
  $PY ~/runs/ledger_event.py "${name}_$(date -u +%Y%m%d)" ~/runs/$name.log status=$s kind=compile task=Q4b-T10 backend=$c \
    seq_len=$S notes="8192 compile${err:+; first error: $err}" > /dev/null
  log "end $name $s $(st $name) ${err:+FIRST_ERROR: $err}"
done
ok=(); for c in "${cfgs[@]}"; do
  case $c in B) d=$Q/traced_libmoe_trn2_229_s${S}_greedy_sweep ;; C) d=$Q/traced_nki_trn2_229_s${S}_greedy_sweep ;;
             V) d=$Q/traced_V_q3final_s$S ;; Eh) d=$Q/traced_Eh_trn2_229_s$S ;; esac
  [ -f $d/model.pt ] && ok+=($c)
done
log "bench configs: ${ok[*]}"
bash $T/bench_q4.sh q4b_t10 $S 3 "${ok[@]}"
log "T10_DONE"
