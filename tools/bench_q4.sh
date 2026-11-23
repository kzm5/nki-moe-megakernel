#!/usr/bin/env bash
# Interleaved TKG benchmark, separate processes, one order per round, same session.
# Usage: bench_q4.sh TAG SEQ ROUNDS CONFIG...   (CONFIG in A B C E Eh V; runs ~/runs/p1_<TAG>_<CONFIG>_s<SEQ>_r<R>)
Q=/home/ubuntu/qwen-30b-a3b
WT=/home/ubuntu/nki-moe-megakernel/.claude/worktrees
COMMON=(--on-device-sampling --top-k 1 --do-sample '' --output-logits)
E_MOD="--moe-fused-nki-kernel-enabled --qwen megakernels.qwen3_moe.qwen_attn_e"
# config seq -> build dir | MK_DIR | env | flags
spec() {
  case $1:$2 in
    A:768) echo "$Q/traced_xla_trn2_229_s768_greedy|$WT/q1q2||" ;;
    B:768) echo "$Q/traced_xlamoe_trn2_229_s768_greedy|$WT/q1q2||--moe-fused-nki-kernel-enabled" ;;
    B:*) echo "$Q/traced_libmoe_trn2_229_s$2_greedy_sweep|$WT/q1q2||--moe-fused-nki-kernel-enabled" ;;
    C:768) echo "$Q/traced_nki_trn2_229_s768_greedy|$WT/q1q2||--enable-nki" ;;
    C:*) echo "$Q/traced_nki_trn2_229_s$2_greedy_sweep|$WT/q1q2||--enable-nki" ;;
    E:*) echo "$Q/traced_E_trn2_229_s$2|$WT/q1q2||$E_MOD" ;;
    Eh:*) echo "$Q/traced_Eh_trn2_229_s$2|$WT/q3/wt/t3|ATTN_E_HOIST=1|$E_MOD" ;;
    V:768) echo "$Q/agent_c1_2|$WT/q3/wt/c1_35309c3|MK_AGENT_VARIANT=1|--enable-nki" ;;
    V:*) echo "$Q/traced_V_q3final_s$2|$WT/q3/wt/c1_35309c3|MK_AGENT_VARIANT=1|--enable-nki" ;;
    *) echo "" ;;
  esac
}
tag=$1; S=$2; rounds=$3; shift 3
L=~/runs/p1/bench_$tag.log
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
exec 9> ~/runs/device.lock; flock 9
for r in $(seq 1 $rounds); do for c in "$@"; do
  IFS='|' read -r cdir mk env flags <<< "$(spec $c $S)"
  name=p1_${tag}_${c}_s${S}_r$r
  [ -f ~/runs/$name/benchmark_report.json ] && continue
  [ -f $cdir/model.pt ] || { log "skip $name (no build)"; continue; }
  # cold page cache makes the IOPS-capped root volume page-fault weights at ~12 MB/s
  for f in $Q/hf_model/model-*.safetensors; do dd if=$f of=/dev/null bs=16M 2>/dev/null; done
  presh=""; [ -d $cdir/weights ] && presh="--save-sharded-checkpoint --skip-sharding"
  df -h / | tail -1 >> $L; log "start $name $presh"
  # shellcheck disable=SC2086
  env MK_DIR=$mk $env bash ~/runs/run_job.sh $name $cdir $flags $presh --seq-len $S "${COMMON[@]}" --skip-compile > /dev/null 2>&1
  bash ~/runs/ledger_bench.sh $name $S $c "$tag interleaved $* (s$S)" >> $L 2>&1
  log "end $name $(grep -o 'EXIT_STATUS=[0-9]*' ~/runs/$name.log | tail -1)"
done; done
log "BENCH_DONE $tag s$S $*"
