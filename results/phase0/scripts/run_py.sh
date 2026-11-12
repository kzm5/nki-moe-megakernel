#!/usr/bin/env bash
# Usage: run_py.sh NAME script.py args...   (device-side python, from the main checkout)
name=$1; shift
. ~/venv-neuron-229/activate-neuron.sh
cd ~/nki-moe-megakernel
log=~/runs/$name.log
date -u +%FT%TZ > ~/runs/$name.start
echo "CMD: python $*" > $log
/usr/bin/time -v python "$@" >> $log 2>&1
echo "EXIT_STATUS=$?" >> $log
echo JOB_DONE >> $log
