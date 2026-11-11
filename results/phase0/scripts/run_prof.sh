#!/usr/bin/env bash
# Usage: run_prof.sh NAME KIND COMPILED_DIR SEQ
name=$1; kind=$2; cdir=$3; seq=$4
. ~/venv-neuron-229/activate-neuron.sh
cd ~/nki-moe-megakernel
log=~/runs/$name.log
date -u +%FT%TZ > ~/runs/$name.start
echo "CMD: python ~/runs/prof/profile_decode.py $kind $cdir ~/runs/prof/$name $seq" > $log
/usr/bin/time -v python ~/runs/prof/profile_decode.py $kind $cdir ~/runs/prof/$name $seq >> $log 2>&1
echo "EXIT_STATUS=$?" >> $log
echo JOB_DONE >> $log
