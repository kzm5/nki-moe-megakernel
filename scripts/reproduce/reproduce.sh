#!/usr/bin/env bash
# Reproduce the trn2 Phase 0/1 results: environment check, builds, frozen validation, interleaved benchmark, figures.
# Usage: reproduce.sh {env|build|validate|bench|figures} [CONFIG_FILE]    (default config: scripts/reproduce/config.sh)
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cmd=${1:?usage: reproduce.sh env|build|validate|bench|figures [config]}
# shellcheck disable=SC1090
source "${2:-$REPO/scripts/reproduce/config.sh}"
FZ=$REPO/tools/validator
CORPUS=$REPO/results/phase1/corpus/corpus.json
COMMON=(--on-device-sampling --top-k 1 --do-sample '' --output-logits)
E_MOD=(--moe-fused-nki-kernel-enabled --qwen megakernels.qwen3_moe.qwen_attn_e)
PY=$VENV/bin/python
mkdir -p "$OUT_ROOT"
L=$OUT_ROOT/reproduce.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

# config -> validator kind | main.py flags | env | source dir
spec() {
  case $1 in
    A) echo "xla||$REPO" ;;
    B) echo "libmoe|--moe-fused-nki-kernel-enabled||$REPO" ;;
    C) echo "nki|--enable-nki||$REPO" ;;
    V) echo "nki|--enable-nki|MK_AGENT_VARIANT=1|$REPO" ;;
    E) echo "E|${E_MOD[*]}||$REPO" ;;
    Eh) [ -n "$EH_DIR" ] || { echo "Eh needs EH_DIR (branch q3/t3-fusion)" >&2; exit 2; }
        echo "E|${E_MOD[*]}|ATTN_E_HOIST=1|$EH_DIR" ;;
    *) echo "unknown config $1" >&2; exit 2 ;;
  esac
}
bdir() { local v="BUILD_$1_$2"; echo "${!v:-$BUILD_ROOT/$1_s$2}"; }
warm() { for f in "$MODEL_PATH"/model-*.safetensors; do dd if="$f" of=/dev/null bs=16M 2>/dev/null; done; }
lock() { exec 9> "$HOME/runs/device.lock"; flock 9; }
main_py() {  # cfg len extra...   (runs main.py evaluate_single from the config's source dir; report -> cwd)
  local c=$1 s=$2; shift 2
  IFS='|' read -r _ flags env src <<< "$(spec "$c")"
  # shellcheck disable=SC2086
  (cd "$src" && source "$VENV/activate-neuron.sh" && env $env python main.py --model qwen3_moe --mode evaluate_single \
     --platform-target trn2 --model-path "$MODEL_PATH" --compiled-model-path "$(bdir "$c" "$s")" --tp-degree 4 \
     --logical-nc-config 2 $flags --seq-len "$s" "${COMMON[@]}" "$@")
}

case $cmd in
env)
  ok=1; chk() { if eval "$2"; then echo "  ok    $1"; else echo "  FAIL  $1"; ok=0; fi; }
  echo "environment:"
  chk "instance trn2.3xlarge" "neuron-ls 2>/dev/null | grep -q trn2.3xlarge"
  chk "LNC=2" "neuron-ls 2>/dev/null | grep -q 'logical-neuroncore-config: 2'"
  chk "venv $VENV" "[ -x $PY ]"
  for p in neuronx-cc=2.24.5133 torch-neuronx=2.9.0.2.13 neuronx-distributed=0.18.27753 \
           neuronx-distributed-inference=0.9.17334 transformers=4.57.6 torch=2.9.0; do
    chk "$p" "$PY -m pip show ${p%%=*} 2>/dev/null | grep -q '^Version: ${p#*=}'"
  done
  chk "nkilib fork 2.29_release @ ddbca35 (editable)" "$PY -c 'import nkilib,os;print(os.path.realpath(nkilib.__file__))' 2>/dev/null | grep -q nki-library"
  chk "aws-neuronx-tools 2.30 (reads NTFF v7)" "dpkg -l aws-neuronx-tools 2>/dev/null | grep -q ' 2.30'"
  chk "HF weights (16 safetensors)" "[ \$(ls $MODEL_PATH/model-*.safetensors 2>/dev/null | wc -l) = 16 ]"
  chk "swap >= 64 GB" "[ \$(free -g | awk '/Swap/{print \$2}') -ge 60 ]"
  chk "RAM >= 120 GiB" "[ \$(free -g | awk '/Mem/{print \$2}') -ge 120 ]"
  chk "free disk >= 30 GB" "[ \$(df -BG --output=avail / | tail -1 | tr -dc 0-9) -ge 30 ]"
  chk "frozen validator hashes (frozen.json)" "$PY $REPO/scripts/reproduce/check_frozen.py $REPO > /dev/null 2>&1"
  chk "validator data ($P1/refs, golden, val/xla_plain)" "[ -d $P1/refs ] && [ -d $P1/golden ] && [ -d $P1/val/xla_plain ]"
  [ $ok = 1 ] && echo "ENV OK" || { echo "ENV FAIL (see docs/REPRODUCE.md)"; exit 1; } ;;
build)
  lock
  for s in $LENGTHS; do for c in $CONFIGS; do
    d=$(bdir "$c" "$s"); [ -f "$d/model.pt" ] && { log "build $c s$s: exists ($d)"; continue; }
    warm; log "build $c s$s -> $d"
    (main_py "$c" "$s" > "$OUT_ROOT/build_${c}_s$s.log" 2>&1); rm -f "$REPO/benchmark_report.json"
    [ -f "$d/model.pt" ] && log "build $c s$s: OK" || log "build $c s$s: FAILED (see $OUT_ROOT/build_${c}_s$s.log)"
  done; done ;;
validate)
  lock
  mkdir -p "$OUT_ROOT/validate"
  for s in $LENGTHS; do
    list=" $CONFIGS "; list=${list// A / }  # XLA (A) first: it is the logits baseline at lengths other than 768
    if [ "$s" != 768 ] || [[ " $CONFIGS " == *" A "* ]]; then list="A $list"; fi
    for c in $list; do
    IFS='|' read -r kind _ env src <<< "$(spec "$c")"
    d=$(bdir "$c" "$s"); o=$OUT_ROOT/validate/${c}_s$s
    [ -f "$d/model.pt" ] || { log "validate $c s$s: no build at $d"; continue; }
    if [ ! -f "$o/device.json" ]; then
      warm; log "validate $c s$s: capture"
      # shellcheck disable=SC2086
      env MK_DIR="$src" $env "$PY" "$FZ/device.py" "$kind" "$d" "$o" --corpus "$CORPUS" --refs "$P1/refs" \
        --tf --ppl --repeat --seq-len "$s" > "$o.log" 2>&1
    fi
    [ -f "$o/device.json" ] || { log "validate $c s$s: capture FAILED (see $o.log)"; continue; }
    base=$P1/val/xla_plain; [ "$s" != 768 ] && base=$OUT_ROOT/validate/A_s$s
    "$PY" "$FZ/validate.py" score --cand "$o" --base-logits "$base" --out "$o/report.json" > "$o.score.log" 2>&1
    log "validate $c s$s: $(grep '^VALIDATE' "$o.score.log" | cut -c10-200)"
    [ -f "${REF_VAL:-/nonexistent}/${c}_s$s/device.json" ] && "$PY" "$REPO/scripts/reproduce/compare_val.py" \
      "$REF_VAL/${c}_s$s" "$o" | tee -a "$L"
  done; done ;;
bench)
  lock
  mkdir -p "$OUT_ROOT/bench"
  for s in $LENGTHS; do for r in $(seq 1 "$N"); do for c in $CONFIGS; do
    o=$OUT_ROOT/bench/${c}_s${s}_r$r; [ -f "$o/benchmark_report.json" ] && continue
    d=$(bdir "$c" "$s"); [ -f "$d/model.pt" ] || { log "bench $c s$s: no build"; continue; }
    presh=(); [ -d "$d/weights" ] && presh=(--save-sharded-checkpoint --skip-sharding)
    warm; mkdir -p "$o"; log "bench $c s$s r$r"
    IFS='|' read -r _ _ _ src <<< "$(spec "$c")"
    (main_py "$c" "$s" --skip-compile "${presh[@]}" > "$o/run.log" 2>&1)
    mv -f "$src/benchmark_report.json" "$o/" 2>/dev/null
    log "bench $c s$s r$r: TKG p50 $("$PY" -c "import json;print(round(json.load(open('$o/benchmark_report.json'))['token_generation_model']['latency_ms_p50'],3))" 2>/dev/null || echo FAILED)"
  done; done; done
  "$PY" "$REPO/scripts/reproduce/summarize.py" "$OUT_ROOT/bench" "$REPO/results/phase1/ledger.jsonl" | tee -a "$L" ;;
figures)
  "$PY" "$REPO/tools/make_figures.py" --out "$REPO/results/final" ;;
*) echo "unknown command $cmd" >&2; exit 2 ;;
esac
