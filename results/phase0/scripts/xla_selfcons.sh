#!/usr/bin/env bash
. ~/venv-neuron-229/activate-neuron.sh
cdir=~/qwen-30b-a3b/traced_xla_trn2_229_s768_greedy
if [ ! -f $cdir/weights/tp3_sharded_checkpoint.safetensors ]; then
  date -u +%FT%TZ > ~/runs/preshard_xla.start
  /usr/bin/time -v python ~/runs/val/preshard.py xla $cdir 768 > ~/runs/preshard_xla.log 2>&1; echo "EXIT_STATUS=$?" >> ~/runs/preshard_xla.log
fi
for r in A B; do
  date -u +%FT%TZ > ~/runs/xla_self_$r.start
  /usr/bin/time -v python ~/runs/val/capture_to.py xla $cdir ~/runs/val/xla_self_$r > ~/runs/xla_self_$r.log 2>&1
  echo "EXIT_STATUS=$?" >> ~/runs/xla_self_$r.log
done
echo XLA_SELF_DONE >> ~/runs/xla_self_B.log
