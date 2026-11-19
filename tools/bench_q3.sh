#!/usr/bin/env bash
# Interleaved TKG benchmark of Q3 variants, separate processes, same session.
# Usage: bench_q3.sh TAG ROUNDS CONFIG...    (runs ~/runs/p1_q3_<TAG>_<CONFIG>_s768_r<R>)
Q=/home/ubuntu/qwen-30b-a3b
WT=/home/ubuntu/nki-moe-megakernel/.claude/worktrees
S=768
COMMON=(--on-device-sampling --top-k 1 --do-sample '' --output-logits)
E_MOD=(--moe-fused-nki-kernel-enabled --qwen megakernels.qwen3_moe.qwen_attn_e)
# config -> build dir | MK_DIR | env | flags
spec() {
  case $1 in
    B) echo "$Q/traced_xlamoe_trn2_229_s768_greedy|$WT/q1q2||--moe-fused-nki-kernel-enabled" ;;
    C) echo "$Q/traced_nki_trn2_229_s768_greedy|$WT/q1q2||--enable-nki" ;;
    E) echo "$Q/traced_E_trn2_229_s768|$WT/q1q2||${E_MOD[*]}" ;;
    Eh) echo "$Q/traced_Eh_trn2_229_s768|$WT/q3/wt/t3|ATTN_E_HOIST=1|${E_MOD[*]}" ;;
    *) echo "" ;;
  esac
}
tag=$1; rounds=$2; shift 2
L=~/runs/p1/bench_q3_$tag.log
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
for r in $(seq 1 $rounds); do for c in "$@"; do
  IFS='|' read -r cdir mk env flags <<< "$(spec $c)"
  name=p1_q3_${tag}_${c}_s${S}_r$r
  [ -f ~/runs/$name/benchmark_report.json ] && continue
  [ -f $cdir/model.pt ] || { log "skip $name (no build)"; continue; }
  # cold page cache makes the IOPS-capped root volume page-fault weights at ~12 MB/s
  for f in $Q/hf_model/model-*.safetensors; do dd if=$f of=/dev/null bs=16M 2>/dev/null; done
  df -h / | tail -1 >> $L; log "start $name"
  # shellcheck disable=SC2086
  env MK_DIR=$mk $env bash ~/runs/run_job.sh $name $cdir $flags --seq-len $S "${COMMON[@]}" --skip-compile > /dev/null 2>&1
  bash ~/runs/ledger_bench.sh $name $S $c "Q3 $tag interleaved $*" >> $L 2>&1
  log "end $name $(grep -o 'EXIT_STATUS=[0-9]*' ~/runs/$name.log | tail -1)"
done; done
log "BENCH_DONE $tag $*"
