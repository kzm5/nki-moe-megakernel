#!/usr/bin/env bash
# Usage: run_cpu.sh NAME python-args...
name=$1; shift
. ~/venv-neuron-229/activate-neuron.sh
log=~/runs/$name.log
date -u +%FT%TZ > ~/runs/$name.start
echo "CMD: python $*" > $log
/usr/bin/time -v python "$@" >> $log 2>&1
echo "EXIT_STATUS=$?" >> $log
echo JOB_DONE >> $log
