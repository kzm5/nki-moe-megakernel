#!/usr/bin/env bash
# Usage: run_job.sh NAME COMPILED_DIR [extra main.py args...]
set -o pipefail
name=$1; cdir=$2; shift 2
log=~/runs/$name.log
source ~/venv-neuron-229/activate-neuron.sh
cd ~/nki-moe-megakernel
date -u +%FT%TZ > ~/runs/$name.start
{ echo "CMD: python main.py --model qwen3_moe --mode evaluate_single --platform-target trn2 --model-path ~/qwen-30b-a3b/hf_model --compiled-model-path $cdir --tp-degree 4 --logical-nc-config 2 $*"; } > $log
/usr/bin/time -v python main.py \
  --model qwen3_moe --mode evaluate_single --platform-target trn2 \
  --model-path ~/qwen-30b-a3b/hf_model --compiled-model-path "$cdir" \
  --tp-degree 4 --logical-nc-config 2 "$@" 2>&1 | tee -a $log
echo "EXIT_STATUS=$?" | tee -a $log
mkdir -p ~/runs/$name
mv -f benchmark_report.json qwen3-30b-a3b-trn2_score_records.csv ~/runs/$name/ 2>/dev/null
echo "JOB_DONE" >> $log
