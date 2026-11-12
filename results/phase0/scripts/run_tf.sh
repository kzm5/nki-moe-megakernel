#!/usr/bin/env bash
# Usage: run_tf.sh NAME KIND COMPILED_DIR OUT [capture_tf.py args...]
name=$1; kind=$2; cdir=$3; out=$4; shift 4
. ~/venv-neuron-229/activate-neuron.sh
cd ~/nki-moe-megakernel
log=~/runs/$name.log
date -u +%FT%TZ > ~/runs/$name.start
echo "CMD: python tools/capture_tf.py $kind $cdir $out $*" > $log
/usr/bin/time -v python ~/nki-moe-megakernel/.claude/worktrees/session-rules/tools/capture_tf.py $kind $cdir $out "$@" >> $log 2>&1
echo "EXIT_STATUS=$?" >> $log
echo JOB_DONE >> $log
