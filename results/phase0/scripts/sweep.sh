#!/usr/bin/env bash
# Seq-len sweep, library MoE vs megakernel, greedy flags without pre-shard. Stops at the 12 h deadline.
DEADLINE=$(date -d "$(cat ~/runs/overnight.start) + 12 hours" +%s)
LAST_START=$((DEADLINE - 37*60))
for s in 256 512 1024 2048; do
  for k in libmoe nki; do
    now=$(date +%s)
    if [ $now -gt $LAST_START ]; then echo "$(date -u +%FT%TZ) SKIP sweep_${k}_s$s (not enough time before deadline)" >> ~/runs/sweep_status.log; continue; fi
    avail=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
    echo "$(date -u +%FT%TZ) START sweep_${k}_s$s disk=${avail}G" >> ~/runs/sweep_status.log
    if [ "$k" = libmoe ]; then extra=--moe-fused-nki-kernel-enabled; else extra=--enable-nki; fi
    timeout --signal=INT $((DEADLINE - now)) bash ~/runs/run_job.sh sweep_${k}_s$s ~/qwen-30b-a3b/traced_${k}_trn2_229_s${s}_greedy_sweep \
      $extra --seq-len $s --on-device-sampling --top-k 1 --do-sample '' --output-logits
    echo "$(date -u +%FT%TZ) END sweep_${k}_s$s rc=$?" >> ~/runs/sweep_status.log
  done
done
echo SWEEP_ALL_DONE > ~/runs/sweep_all.done
