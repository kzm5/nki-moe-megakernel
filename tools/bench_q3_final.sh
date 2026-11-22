#!/usr/bin/env bash
# Q3 final: interleaved TKG benchmark of B, C and one collectives variant V, separate processes, same session.
# Usage: V_WT=<worktree> V_768=<768 build> bench_q3_final.sh compile SEQ... | rounds SEQ N
# V at other lengths is compiled into $Q/traced_V_q3final_s<SEQ> (compile runs are not counted).
Q=/home/ubuntu/qwen-30b-a3b
Q1=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2
L=~/runs/p1/bench_q3_final.log
FLAGS=(--on-device-sampling --top-k 1 --do-sample '' --output-logits)
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
cdir() {  # config seq
  case $1:$2 in
    B:768) echo $Q/traced_xlamoe_trn2_229_s768_greedy ;;
    B:*) echo $Q/traced_libmoe_trn2_229_s$2_greedy_sweep ;;
    C:768) echo $Q/traced_nki_trn2_229_s768_greedy ;;
    C:*) echo $Q/traced_nki_trn2_229_s$2_greedy_sweep ;;
    V:768) echo $V_768 ;;
    V:*) echo $Q/traced_V_q3final_s$2 ;;
  esac
}
env_of() {  # config -> MK_DIR and env
  case $1 in
    B) echo "MK_DIR=$Q1" ;;
    C) echo "MK_DIR=$Q1" ;;
    V) echo "MK_DIR=$V_WT MK_AGENT_VARIANT=1" ;;
  esac
}
flags_of() { case $1 in B) echo --moe-fused-nki-kernel-enabled ;; *) echo --enable-nki ;; esac; }
warm() { for f in $Q/hf_model/model-*.safetensors; do dd if=$f of=/dev/null bs=16M 2>/dev/null; done; }
exec 9> ~/runs/device.lock; flock 9
case $1 in
  compile)
    shift
    for s in "$@"; do
      d=$(cdir V $s); [ -f $d/model.pt ] && continue
      name=p1_q3final_compile_V_s$s; warm; df -h / | tail -1 >> $L; log "compile $name"
      # shellcheck disable=SC2046
      env $(env_of V) bash ~/runs/run_job.sh $name $d $(flags_of V) --seq-len $s "${FLAGS[@]}" > /dev/null 2>&1
      log "compiled $name $(grep -o 'EXIT_STATUS=[0-9]*' ~/runs/$name.log | tail -1)"
    done ;;
  rounds)
    s=$2; n=$3
    for r in $(seq 1 $n); do for c in B C V; do
      name=p1_q3final_${c}_s${s}_r$r; d=$(cdir $c $s)
      [ -f ~/runs/$name/benchmark_report.json ] && continue
      [ -f $d/model.pt ] || { log "skip $name (no build)"; continue; }
      presh=""; [ -d $d/weights ] && presh="--save-sharded-checkpoint --skip-sharding"
      warm; df -h / | tail -1 >> $L; log "start $name $presh"
      # shellcheck disable=SC2046,SC2086
      env $(env_of $c) bash ~/runs/run_job.sh $name $d $(flags_of $c) $presh --seq-len $s "${FLAGS[@]}" --skip-compile > /dev/null 2>&1
      bash ~/runs/ledger_bench.sh $name $s $c "Q3 final interleaved B/C/V (V=$V_768)" >> $L 2>&1
      log "end $name $(grep -o 'EXIT_STATUS=[0-9]*' ~/runs/$name.log | tail -1)"
    done; done ;;
esac
log "DONE $*"
