#!/usr/bin/env bash
# T8: interleaved B -> C -> E decode benchmarks, separate processes, same session.
# Usage: bench_bce.sh warmup | rounds SEQ N     (warmup compiles missing builds; its runs are not counted)
Q=/home/ubuntu/qwen-30b-a3b
COMMON=(--on-device-sampling --top-k 1 --do-sample '' --output-logits --skip-compile)
cdir() {  # config seq
  case $1:$2 in
    B:768) echo $Q/traced_xlamoe_trn2_229_s768_greedy ;;
    B:*) echo $Q/traced_libmoe_trn2_229_s$2_greedy_sweep ;;
    C:768) echo $Q/traced_nki_trn2_229_s768_greedy ;;
    C:*) echo $Q/traced_nki_trn2_229_s$2_greedy_sweep ;;
    E:*) echo $Q/traced_E_trn2_229_s$2 ;;
  esac
}
flags() {
  case $1 in
    B) echo --moe-fused-nki-kernel-enabled ;;
    C) echo --enable-nki ;;
    E) echo --moe-fused-nki-kernel-enabled --qwen megakernels.qwen3_moe.qwen_attn_e ;;
  esac
}
one() {  # name config seq [extra]
  local name=$1 c=$2 s=$3; shift 3
  [ -f ~/runs/$name/benchmark_report.json ] && return
  df -h / | tail -1 >> ~/runs/p1/bench.log
  echo "$(date -u +%FT%TZ) start $name" >> ~/runs/p1/bench.log
  # shellcheck disable=SC2046
  bash ~/runs/run_job.sh $name $(cdir $c $s) $(flags $c) --seq-len $s "${COMMON[@]}" "$@" > /dev/null 2>&1
  bash ~/runs/ledger_bench.sh $name $s $c "T8 interleaved B/C/E" >> ~/runs/p1/bench.log 2>&1
  echo "$(date -u +%FT%TZ) end $name $(grep -o 'EXIT_STATUS=[0-9]*' ~/runs/$name.log | tail -1)" >> ~/runs/p1/bench.log
}
case $1 in
  warmup)
    for s in 2048 4096; do
      [ -f $(cdir E $s)/model.pt ] && continue
      name=p1_compile_E_s$s
      echo "$(date -u +%FT%TZ) compile E s$s" >> ~/runs/p1/bench.log
      # shellcheck disable=SC2046
      bash ~/runs/run_job.sh $name $(cdir E $s) $(flags E) --seq-len $s --on-device-sampling --top-k 1 --do-sample '' --output-logits > /dev/null 2>&1
      echo "$(date -u +%FT%TZ) compiled E s$s $(grep -o 'EXIT_STATUS=[0-9]*' ~/runs/$name.log | tail -1)" >> ~/runs/p1/bench.log
    done ;;
  rounds)
    s=$2; n=$3
    for r in $(seq 1 $n); do for c in B C E; do one p1_bench_${c}_s${s}_r$r $c $s; done; done ;;
esac
echo "$(date -u +%FT%TZ) BENCH_DONE $*" >> ~/runs/p1/bench.log
