#!/usr/bin/env bash
# Noise floor, 5 runs per config, rotating pre-sharded weights (disk fits one set).
del() {
  d=~/qwen-30b-a3b/$1/weights; s=$(du -sh $d | cut -f1); rm -rf $d
  echo "$(date -u +%FT%TZ) deleted $d ($s pre-sharded weights; noise runs for this config done)" >> ~/runs/deletions.log
}
bash ~/runs/noise10.sh xla 5
del traced_xla_trn2_229_s768_greedy
bash ~/runs/noise10.sh libmoe 5
del traced_xlamoe_trn2_229_s768_greedy
bash ~/runs/noise10.sh nki 5
del traced_nki_trn2_229_s768_greedy
echo NOISE5_ALL_DONE >> ~/runs/noise10_status.log
