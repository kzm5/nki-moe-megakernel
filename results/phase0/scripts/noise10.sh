#!/usr/bin/env bash
# Usage: noise10.sh KIND N   (KIND: xla|libmoe|nki). Pre-shards the s768 greedy build, runs N benchmarks, deletes shards.
kind=$1; n=${2:-10}
case $kind in
  xla) cdir=~/qwen-30b-a3b/traced_xla_trn2_229_s768_greedy; extra=() ;;
  libmoe) cdir=~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_greedy; extra=(--moe-fused-nki-kernel-enabled) ;;
  nki) cdir=~/qwen-30b-a3b/traced_nki_trn2_229_s768_greedy; extra=(--enable-nki) ;;
esac
. ~/venv-neuron-229/activate-neuron.sh
if [ ! -f $cdir/weights/tp3_sharded_checkpoint.safetensors ]; then
  date -u +%FT%TZ > ~/runs/preshard_$kind.start
  /usr/bin/time -v python ~/runs/val/preshard.py $kind $cdir 768 > ~/runs/preshard_$kind.log 2>&1
  echo "EXIT_STATUS=$?" >> ~/runs/preshard_$kind.log
fi
for i in $(seq 1 $n); do
  [ -f ~/runs/noise10_${kind}_r$i/benchmark_report.json ] && continue
  bash ~/runs/run_job.sh noise10_${kind}_r$i $cdir "${extra[@]}" --seq-len 768 --on-device-sampling --top-k 1 \
    --do-sample '' --output-logits --save-sharded-checkpoint --skip-compile
done
echo "NOISE10_${kind}_DONE" >> ~/runs/noise10_status.log
