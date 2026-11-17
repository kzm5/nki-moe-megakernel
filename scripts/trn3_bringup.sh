#!/usr/bin/env bash
# trn3 bring-up for the one-day B/C/E comparison (not yet run). See scripts/trn3_bringup.md.
# Usage: scripts/trn3_bringup.sh STAGE    (check | compile | bench | profile | validate | all)
set -euo pipefail
STAGE=${1:-check}
REPO=$(cd "$(dirname "$0")/.." && pwd)
Q=~/qwen-30b-a3b
R=~/runs/trn3
VENV=${VENV:-~/venv-neuron-229}
SEQS=(768 2048 4096)
mkdir -p "$R"
. "$VENV/activate-neuron.sh"
cd "$REPO"

common=(--model qwen3_moe --platform-target trn3 --model-path "$Q/hf_model" --tp-degree 4 --logical-nc-config 2
        --on-device-sampling --top-k 1 --do-sample "" --output-logits)
declare -A FLAGS=([A]="" [B]="--moe-fused-nki-kernel-enabled" [C]="--enable-nki"
                  [E]="--moe-fused-nki-kernel-enabled --qwen megakernels.qwen3_moe.qwen_attn_e")
build() { echo "$Q/traced_$1_trn3_s$2"; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/bringup.log"; }

check() {
  neuron-ls | tee "$R/neuron-ls.txt"
  grep -q trn3 "$R/neuron-ls.txt" || { log "FAIL: not a trn3 instance"; exit 1; }
  dpkg -l | grep -i neuron | awk '{print $2, $3}' | tee "$R/neuron-pkgs.txt"
  python -c "import neuronxcc, torch_neuronx, neuronx_distributed_inference as n; print(neuronxcc.__version__, torch_neuronx.__version__)" | tee -a "$R/bringup.log"
  neuronx-cc compile --help | grep -q "trn3" || { log "FAIL: compiler has no trn3 target"; exit 1; }
  python -c "import nkilib, nki_kernels._legacy.attention, nki_kernels._legacy.moe" && log "nkilib imports OK"
  df -h / | tail -1; free -g | sed -n 2p
  log "check OK"
}

compile_one() {  # config seq
  local c=$1 s=$2 d; d=$(build "$c" "$s")
  [ -f "$d/model.pt" ] && { log "skip compile $c s$s"; return; }
  log "compile $c s$s"
  # shellcheck disable=SC2086
  python main.py "${common[@]}" --mode generate --seq-len "$s" --compiled-model-path "$d" ${FLAGS[$c]} \
    --prompts "I believe the meaning of life is" > "$R/compile_${c}_s$s.log" 2>&1 || { log "FAIL compile $c s$s"; return 1; }
}

compile() {  # A only at 768 (validator baseline); B, C, E at every seq
  compile_one A 768
  for s in "${SEQS[@]}"; do for c in B C E; do compile_one "$c" "$s" || true; done; done
}

bench() {  # interleaved rounds B -> C -> E, separate processes; n=5 at 768, n=3 at 2048/4096
  for s in "${SEQS[@]}"; do
    n=3; [ "$s" = 768 ] && n=5
    for r in $(seq 1 "$n"); do for c in B C E; do
      d=$(build "$c" "$s"); [ -f "$d/model.pt" ] || continue
      log "bench $c s$s r$r"
      # evaluate_single on trn3 also compiles an XLA baseline for logit_validation; use benchmark mode
      # through tools/bench_only.py (same NxDI benchmark_sampling call as main.py's trn2 branch).
      # shellcheck disable=SC2086
      python tools/bench_only.py "${common[@]}" --seq-len "$s" --compiled-model-path "$d" ${FLAGS[$c]} \
        --skip-compile --out "$R/bench_${c}_s${s}_r$r.json" > "$R/bench_${c}_s${s}_r$r.log" 2>&1 || log "FAIL bench $c s$s r$r"
    done; done
  done
}

profile() {  # one per-layer decode profile per config at 768 (largest bucket), same method as Phase 0
  for c in B C E; do
    log "profile $c"
    bash ~/runs/prof/run_prof.sh "trn3_prof_${c}_s768" "$c" "$(build "$c" 768)" 768 || log "FAIL profile $c"
  done
}

validate() {  # frozen validator: XLA trn3 baseline first, then B, C, E (golden is hardware-independent)
  V=tools/validator
  python $V/device.py xla "$(build A 768)" "$R/val/xla_base" --corpus results/phase1/corpus/corpus.json \
    --refs ~/runs/p1/refs --tf --ppl --repeat
  for c in B C E; do
    kind=$([ $c = B ] && echo libmoe || ([ $c = C ] && echo nki || echo E))
    python $V/device.py "$kind" "$(build "$c" 768)" "$R/val/$c" --corpus results/phase1/corpus/corpus.json \
      --refs ~/runs/p1/refs --tf --ppl --repeat
    python $V/validate.py score --cand "$R/val/$c" --base "$R/val/xla_base" --out "$R/val/report_$c.json"
  done
}

case $STAGE in
  check) check ;; compile) compile ;; bench) bench ;; profile) profile ;; validate) validate ;;
  all) check; compile; bench; profile; validate ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
