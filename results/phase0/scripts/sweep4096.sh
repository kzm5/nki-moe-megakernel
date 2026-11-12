#!/usr/bin/env bash
G=(--seq-len 4096 --on-device-sampling --top-k 1 --do-sample '' --output-logits)
for k in libmoe nki; do
  case $k in libmoe) f=--moe-fused-nki-kernel-enabled ;; nki) f=--enable-nki ;; esac
  name=sweep_${k}_s4096
  echo "$(date -u +%FT%TZ) START $name disk=$(df -h --output=avail / | tail -1)" >> ~/runs/sweep4096_status.log
  bash ~/runs/run_job.sh $name ~/qwen-30b-a3b/traced_${k}_trn2_229_s4096_greedy_sweep $f "${G[@]}"
  echo "$(date -u +%FT%TZ) END $name $(grep -o 'EXIT_STATUS=[0-9]*' ~/runs/$name.log | tail -1)" >> ~/runs/sweep4096_status.log
done
echo SWEEP4096_DONE >> ~/runs/sweep4096_status.log
