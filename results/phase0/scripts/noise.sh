#!/usr/bin/env bash
for i in 1 2 3 4 5; do
  bash ~/runs/run_job.sh noise_libmoe_r$i ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_greedy --moe-fused-nki-kernel-enabled --seq-len 768 --on-device-sampling --top-k 1 --do-sample '' --output-logits --save-sharded-checkpoint --skip-compile
  bash ~/runs/run_job.sh noise_nki_r$i ~/qwen-30b-a3b/traced_nki_trn2_229_s768_greedy --enable-nki --seq-len 768 --on-device-sampling --top-k 1 --do-sample '' --output-logits --save-sharded-checkpoint --skip-compile
done
echo NOISE_ALL_DONE > ~/runs/noise_all.done
