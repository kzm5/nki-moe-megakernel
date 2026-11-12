#!/usr/bin/env bash
# Usage: PRESHARD=0|1 run_replay.sh NAME KIND COMPILED_DIR OUT_DIR
name=$1; kind=$2; cdir=$3; out=$4
. ~/venv-neuron-229/activate-neuron.sh
cd ~/nki-moe-megakernel
log=~/runs/$name.log
date -u +%FT%TZ > ~/runs/$name.start
echo "CMD: PRESHARD=$PRESHARD python ~/runs/val/replay_validate.py $kind $cdir $out" > $log
/usr/bin/time -v python ~/runs/val/replay_validate.py $kind $cdir $out >> $log 2>&1
echo "EXIT_STATUS=$?" >> $log
echo JOB_DONE >> $log
