#!/usr/bin/env bash
# Config D: library MoE + NxDI NKI attention-block TKG kernel. 1 compile+bench and 2 reruns per seq_len.
D=(--moe-fused-nki-kernel-enabled --attn-block-tkg-nki-kernel-enabled --attn-block-tkg-nki-kernel-cascaded-attention
   --qkv-kernel-enabled --on-device-sampling --top-k 1 --do-sample '' --output-logits)
for s in ${SEQS:-768 2048}; do
  cdir=~/qwen-30b-a3b/traced_cfgD_trn2_229_s${s}_greedy
  echo "$(date -u +%FT%TZ) START cfgD_s${s}_r1 disk=$(df -h --output=avail / | tail -1)" >> ~/runs/cfgD_status.log
  bash ~/runs/run_job.sh cfgD_s${s}_r1 $cdir "${D[@]}" --seq-len $s
  rc=$(grep -o 'EXIT_STATUS=[0-9]*' ~/runs/cfgD_s${s}_r1.log | tail -1)
  echo "$(date -u +%FT%TZ) END cfgD_s${s}_r1 $rc" >> ~/runs/cfgD_status.log
  [ "$rc" = "EXIT_STATUS=0" ] || continue
  for r in 2 3; do
    bash ~/runs/run_job.sh cfgD_s${s}_r$r $cdir "${D[@]}" --seq-len $s --skip-compile
    echo "$(date -u +%FT%TZ) END cfgD_s${s}_r$r $(grep -o 'EXIT_STATUS=[0-9]*' ~/runs/cfgD_s${s}_r$r.log | tail -1)" >> ~/runs/cfgD_status.log
  done
done
echo CFGD_ALL_DONE >> ~/runs/cfgD_status.log
