#!/usr/bin/env bash
# T8b at seq 768: N = B + one null NKI call per layer, F = E with C's compiler flags; interleaved with B, C, E.
# Usage: bench_t8b.sh warmup | rounds N
Q=/home/ubuntu/qwen-30b-a3b
S=768
COMMON=(--on-device-sampling --top-k 1 --do-sample '' --output-logits)
cdir() {
  case $1 in
    B) echo $Q/traced_xlamoe_trn2_229_s768_greedy ;;
    C) echo $Q/traced_nki_trn2_229_s768_greedy ;;
    E) echo $Q/traced_E_trn2_229_s768 ;;
    N) echo $Q/traced_Bnull_trn2_229_s768 ;;
    F) echo $Q/traced_Ecflags_trn2_229_s768 ;;
  esac
}
flags() {
  case $1 in
    B) echo --moe-fused-nki-kernel-enabled ;;
    C) echo --enable-nki ;;
    E) echo --moe-fused-nki-kernel-enabled --qwen megakernels.qwen3_moe.qwen_attn_e ;;
    N) echo --moe-fused-nki-kernel-enabled --qwen megakernels.qwen3_moe.qwen_null_nki ;;
    F) echo --moe-fused-nki-kernel-enabled --qwen megakernels.qwen3_moe.qwen_attn_e_cflags ;;
  esac
}
log() { echo "$(date -u +%FT%TZ) $*" >> ~/runs/p1/bench_t8b.log; }
case $1 in
  warmup)
    for c in N F; do
      [ -f $(cdir $c)/model.pt ] && continue
      name=p1_compile_${c}_s$S; log "compile $c"
      # shellcheck disable=SC2046
      bash ~/runs/run_job.sh $name $(cdir $c) $(flags $c) --seq-len $S "${COMMON[@]}" > /dev/null 2>&1
      log "compiled $c $(grep -o 'EXIT_STATUS=[0-9]*' ~/runs/$name.log | tail -1)"
    done ;;
  rounds)
    for r in $(seq 1 $2); do for c in B C E N F; do
      name=p1_t8b_${c}_s${S}_r$r
      [ -f ~/runs/$name/benchmark_report.json ] && continue
      [ -f $(cdir $c)/model.pt ] || { log "skip $name (no build)"; continue; }
      df -h / | tail -1 >> ~/runs/p1/bench_t8b.log; log "start $name"
      # shellcheck disable=SC2046
      bash ~/runs/run_job.sh $name $(cdir $c) $(flags $c) --seq-len $S "${COMMON[@]}" --skip-compile > /dev/null 2>&1
      bash ~/runs/ledger_bench.sh $name $S $c "T8b interleaved B/C/E/N/F" >> ~/runs/p1/bench_t8b.log 2>&1
      log "end $name $(grep -o 'EXIT_STATUS=[0-9]*' ~/runs/$name.log | tail -1)"
    done; done ;;
esac
log "T8B_DONE $*"
