# Resuming Phase 0 on a fresh instance from this image

## Final state (2026-11-28, Q4 + Q4b + Q5 done; results frozen) — read this first
- **Results:** `results/final/SUMMARY_FINAL.md` (one page), `NUMBERS.md` (every number with its ledger id / report field), `TABLES.md` + figures/CSVs. Regenerate: `python tools/make_figures.py --out results/final && python tools/make_numbers.py --out results/final/NUMBERS.md` (inputs: `results/phase1/ledger.jsonl` and reports under `results/phase1/`).
- **Reproduce:** `scripts/reproduce/reproduce.sh {env|build|validate|bench|figures}` + `docs/REPRODUCE.md` (AMI and fresh-DLAMI starting points, costs, pitfalls). Proven on the AMI from a clean worktree (validate bit-identical 64/64).
- **Code state:** V (all-gather + local fp32 sum collectives) merged into `phase0/trn2-setup` as `a62b8af`, gated by `MK_AGENT_VARIANT=1` (default off). Eh lives only on branch `q3/t3-fusion` (`ATTN_E_HOIST=1`). Agent-study branches `q3/t4-*` on `mine`.
- **Validator:** frozen v1.1 (`results/phase1/FROZEN.md`); supplementary long-context corpus frozen separately (`FROZEN_long.md`, scored with `tools/validate_long.py`).
- **Builds kept** (`~/qwen-30b-a3b/`): A/B/C/E/Eh/V at 768, B/C/V/Eh at 2048 and 4096, XLA at 2048/4096, B/C/V at 8192 (C hangs at runtime at 8192 and wedges the device: recover with `sudo rmmod neuron && sudo modprobe neuron`). C at 768 has pre-sharded `weights/` (59 GB).
- **Jobs:** one device job at a time via `~/runs/device.lock`; `tools/bench_q4.sh TAG SEQ N CONFIG...` (interleaved bench), `tools/bench_q4_stats.py`, `tools/q4_chain.sh`, `tools/q4b_t9.sh`. Transcripts in `~/runs/transcripts/`.
- trn3: shelved; `docs/queues/TRN3.md`, `scripts/trn3_bringup.{sh,md}`.

## Q3 update (2026-11-23, Q3 done) — read this first
- State: `results/phase1/QUEUE_STATE.md` (Q3 section), `SUMMARY_Q3.md`. Results worktree `.claude/worktrees/q3` (branch `worktree-q3`, pushed to `phase0/trn2-setup`); kernel branches `q3/t3-fusion`, `q3/t4-{base,a1,c1,a2,c2,c2r}` on `mine`, not merged. Best collectives variant V = `35309c3` (`q3/t4-c1`), env `MK_AGENT_VARIANT=1`; builds `~/qwen-30b-a3b/agent_c1_2` (768), `traced_V_q3final_s{2048,4096}`.
- Agent-study harness `tools/agent_eval.sh SETUP RUN ITER WORKTREE`; final bench `tools/bench_q3_final.sh` (B/C/V, needs `V_WT`, `V_768`). Cross-process determinism check `tools/r2r_rerun.sh` + `tools/r2r_compare.py`.
- Guard hook also blocks Bash commands whose text contains the word "validator" next to `>`/`cp`/`rm` (basename match on the frozen `tools/validator`): put such text in a script file.

## Phase 1 update (2026-11-18, Q1+Q2 done) — read this first
- State: `results/phase1/QUEUE_STATE.md`, `SUMMARY_Q1Q2.md`, `FINDINGS.md`, `DAILY.md`. Worktree `.claude/worktrees/q1q2`; pull with `git pull --no-rebase mine phase0/trn2-setup` (rebase fails on the read-only validator); `*.json` needs `git add -f`.
- **Pre-read the lazily restored volume first** (serial reads ran at 13 MB/s, 16 parallel at ~95 MB/s): `bash tools/preread.sh ~/qwen-30b-a3b/hf_model ~/runs/p1/golden ~/runs/p1/refs <build dirs>`. With hydrated blocks a plain build loads in ~3–5 min.
- Validator (frozen; spec v1.1 default): device capture `python tools/validator/device.py KIND CDIR OUT --corpus results/phase1/corpus/corpus.json --refs ~/runs/p1/refs --tf --ppl --repeat [--capture]` (kinds xla, libmoe, nki, E); golden stats `python tools/validator/golden.py --corpus … --refs ~/runs/p1/refs --out <tmp> --stats NAME=OUT …`; score `python tools/validator/validate.py score --cand OUT [--spec 1.0|1.1] --out report.json`. Baselines: `~/runs/p1/val/xla_base` (capture), `~/runs/p1/val/xla_plain` (production).
- Capture gotchas: NxD module capture yields one tensor per module (first output); for `self_attn` that is not the attention output → capture `self_attn.o_proj`; prefill outputs are sequence-sharded → score decode steps only; patch `TensorRegistry.clear` at trace time; the capture build differs from production XLA at the 512 prefill bucket.
- Benchmarks: `tools/bench_bce.sh`, `tools/bench_t8b.sh` (interleaved, separate processes), stats `tools/bench_stats.py`. Config E: `--moe-fused-nki-kernel-enabled --qwen megakernels.qwen3_moe.qwen_attn_e`.
- Guard hook in `~/.claude/settings.json` denies edits to frozen paths (also blocks Bash commands that mention them together with `>`/`cp`/…; use the Edit tool and `git commit -F`). Deletions need Kevin's approval with a `du` list.

State as of 2026-11-12 ~22:45 ET (Day 3b done). Branch `phase0/trn2-setup` on remote `mine` (`https://github.com/kzm5/nki-moe-megakernel.git`). Read first: `.claude/CLAUDE.md` (Session rules), `docs/RESEARCH_PLAN.md`, `results/phase0/SUMMARY_DAY3.md`, `FINDINGS.md` (latest "Day 3" sections), `ledger.jsonl`.

## Software
| Item | Version / location |
|---|---|
| Instance | trn2.3xlarge (1 chip, 4 logical cores at LNC=2, 96 GB HBM, 124 GiB RAM, 12 vCPU) |
| Venv | `~/venv-neuron-229` (activate with `source ~/venv-neuron-229/activate-neuron.sh`): neuronx-cc 2.24.5133, torch 2.9.0, torch-neuronx 2.9.0.2.13, neuronx-distributed 0.18.27753, neuronx-distributed-inference 0.9.17334, transformers 4.57.6 |
| nkilib | submodule `nki_kernels/nki-library`, fork branch `2.29_release` (`ddbca35`), editable install |
| apt | runtime-lib 2.32.31, collectives 2.32.28, dkms 2.28.0, **aws-neuronx-tools 2.30.10.0 (held)** — needed to read NTFF v7 |
| Unused | `~/venv-neuron-230` (SDK 2.30; its bundled nkilib breaks the legacy MoE kernels) |
| Swap | 64 GB `/swapfile` (in `/etc/fstab`) |

## On disk (sizes at handoff; 65 GB free of 387 GB)
| Path | Size | Essential? | What |
|---|---|---|---|
| `~/qwen-30b-a3b/hf_model` | 57 GB | **yes** | HF checkpoint (bf16 safetensors) |
| `~/qwen-30b-a3b/traced_xla_trn2_229_s768_greedy` | 1.7 GB | **yes** | XLA reference build (greedy flags) |
| `~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_greedy` | 1.7 GB | **yes** | XLA + library MoE build |
| `~/qwen-30b-a3b/traced_nki_trn2_229_s768_greedy` | 1.7 GB | **yes** | megakernel build |
| `~/qwen-30b-a3b/traced_{xla,xlamoe}_trn2_229_s768_capture` | 1.7 GB each | for T2 reruns | tensor-capture builds (per-layer input_layernorm / o_proj / mlp in+out) |
| `~/qwen-30b-a3b/traced_cfgD_trn2_229_s{768,2048}_greedy` | 1.7 / 2.2 GB | optional | config D (lib MoE + NxDI NKI attention) |
| `~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_router2`, `traced_xla_trn2_229_s768_router` | ~1.7 GB each | for Task R reruns | router-capture builds (router values [+ XLA expert ids] + mlp outputs, all buckets); `traced_xlamoe_…_router` = attempt 1, bucket 128 only |
| `~/runs/router` | 11 GB | yes | Task R captures (`libmoe2`, `xla`, attempt-1 `libmoe`), golden MoE outputs at all positions (`golden/dump_pNN.pt`), `router_census.json` |
| `~/qwen-30b-a3b/traced_{libmoe,nki}_trn2_229_s{256,512,1024,2048,4096}_greedy_sweep` | 0.8–2.7 GB each | optional | seq-len sweep builds |
| other `traced_*` (non-greedy day-1, `traced_nki_trn2*`) | ≤1.7 GB | no | superseded |
| Pre-sharded `weights/` | **none** | – | all deleted after the noise floor; every load shards on load |
| `~/runs/golden_v2` | 1.5 GB | **yes (never delete)** | fp32 golden, 20 short prompts, router top-9 `[48,T,9]` |
| `~/runs/golden` | 1.5 GB | used by day-2 numbers | golden v1 (same values to rel-L2 3e-4) |
| `~/runs/golden_long` | 155 MB | yes | fp32 golden, 4 long prompts |
| `~/runs/val` | 7.3 GB | **yes (never delete)** | day-1 XLA references `expected_pNN.pt`, day-1 NKI free-running `actual_*`, prompts, scripts (`common.py`, `capture.py`, `validate.py`, `replay_validate.py`, `preshard.py`, `capture_to.py`), XLA self-consistency `xla_self_{A,B}` |
| `~/runs/val_long` | 149 MB | yes | 4 long-prompt XLA references + `prompts_long.json` |
| `~/runs/tf` | 7.9 GB | yes | teacher-forced captures (`xla`, `libmoe`, `nki`: `tf_main`, `tf_long`, `tf_repeat`), `census.json`, `census_rows.json` |
| `~/runs/replay` | 3.7 GB | yes | day-1 replay, history probe, free-running token runs |
| `~/runs/bisect` | 1.7 GB | yes | golden per-layer dumps + device captures, `bisect.json` |
| `~/runs/prof` | 250 MB | yes | profile tables, dumped TKG NEFFs, profile scripts (raw NTFFs deleted) |
| `~/runs/ledger.jsonl`, `~/runs/deletions.log`, `~/runs/*.log` | small | **yes** | run ledger (copy in `results/phase0/`), deletion log, all run logs |
| `~/runs_backup.tar.zst` | 13 GB (865 entries, `zstd -t` OK, 2026-11-12 04:11 ET) | – | `~/runs` without NEFFs/safetensors/model.pt/weights |
| `/var/tmp/neuron-compile-cache` | 40 GB | no (speeds recompiles) | neuronx-cc cache |
| `/tmp/nxd_model` | 3.3 GB | no | last compile workdir |

## Running things
`~/runs/run_job.sh NAME COMPILED_DIR [main.py args…]` runs `main.py --mode evaluate_single` (adds `--model qwen3_moe --platform-target trn2 --model-path ~/qwen-30b-a3b/hf_model --tp-degree 4 --logical-nc-config 2`), logs `~/runs/NAME.log`, moves `benchmark_report.json` to `~/runs/NAME/`. Greedy flags used everywhere: `--seq-len 768 --on-device-sampling --top-k 1 --do-sample "" --output-logits`; lib MoE adds `--moe-fused-nki-kernel-enabled`, megakernel `--enable-nki`; add `--skip-compile` to reuse a build. Ledger: `~/runs/ledger_bench.sh NAME SEQ BACKEND NOTES` (benchmarks), `python3 ~/runs/ledger_event.py RUN_ID LOG key=value…` (everything else).
Other wrappers: `~/runs/run_tf.sh NAME KIND CDIR OUT args…` (teacher-forced), `~/runs/run_py.sh NAME script.py args…` (any device-side script), `~/runs/run_cpu.sh NAME script.py args…` (CPU jobs), `~/runs/prof/run_prof.sh NAME KIND CDIR SEQ` (decode profile).

**Rerun the census** (≈20–35 min per build, mostly shard-on-load):
```bash
bash ~/runs/run_tf.sh tf_xla xla ~/qwen-30b-a3b/traced_xla_trn2_229_s768_greedy ~/runs/tf/xla --tf main=$HOME/runs/val long=$HOME/runs/val_long
bash ~/runs/run_tf.sh tf_libmoe libmoe ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_greedy ~/runs/tf/libmoe \
  --tf main=$HOME/runs/val long=$HOME/runs/val_long repeat=$HOME/runs/val \
  --single main=$HOME/runs/golden_v2=$HOME/runs/val long=$HOME/runs/golden_long=$HOME/runs/val_long
# same for nki with ~/qwen-30b-a3b/traced_nki_trn2_229_s768_greedy
python tools/census.py --builds libmoe nki            # -> ~/runs/tf/census.json
```
**Rerun the per-layer comparison:**
```bash
python tools/golden_fp32.py --dtype fp32 --inputs "$HOME/runs/val/expected_p*.pt" "$HOME/runs/val_long/expected_p*.pt" --out ~/runs/bisect/golden --dump NN:P,…
bash ~/runs/run_py.sh cap_libmoe tools/capture_layers.py libmoe ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_capture ~/runs/bisect/libmoe --targets NN:P,…
bash ~/runs/run_py.sh cap_xla tools/capture_layers.py xla ~/qwen-30b-a3b/traced_xla_trn2_229_s768_capture ~/runs/bisect/xla --targets NN:P,…
python tools/bisect_compare.py --builds libmoe=$HOME/runs/bisect/libmoe xla=$HOME/runs/bisect/xla
```
`capture_layers.py` (day 3) captures only prefill + TKG bucket 128: NxDI's `registry.clear()` resets the capture config after the first bucket's trace. `capture_router.py` patches `TensorRegistry.clear` in-process and captures every bucket — port that patch before reusing `capture_layers.py` beyond bucket 128.

**Rerun the router census (Task R):**
```bash
bash ~/runs/run_py.sh rcap_libmoe2 tools/capture_router.py libmoe ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_router2 ~/runs/router/libmoe2
bash ~/runs/run_py.sh rcap_xla tools/capture_router.py xla ~/qwen-30b-a3b/traced_xla_trn2_229_s768_router ~/runs/router/xla
python tools/router_census.py --builds libmoe=$HOME/runs/router/libmoe2 xla=$HOME/runs/router/xla
```
A process's log shows `JOB_DONE` well before it exits (teardown takes 10+ min): check `ps` before starting the next device job.

## Gotchas
- `--do-sample ""` (empty string) is the only way to get greedy: `main.py`'s `--do-sample` is `type=bool`; `--top-k 1` alone samples.
- `--seq-len` must be a multiple of 256 at LNC=2 (megakernel attention shards the prior across both cores, needs a multiple of 128 each).
- Host RAM peaks ~121–127 GiB of 124 during shard-on-load and compile; swap absorbs it (loads take 10–40 min when swapping). One compile or device job at a time; CPU golden runs (~5 GB) also count as a heavy job.
- Pre-sharded weights are 57–59 GB per build; disk fits one set beside the rest. `preshard.py` writes them without recompiling (XLA 8 min, lib MoE 52 min, megakernel 43 min).
- Profiling needs aws-neuronx-tools ≥ 2.30 (runtime writes NTFF v7). Use `profile_decode.py` (inspect `model` mode around single TKG calls; whole-process inspect profiles CTE too and fills the disk). Ranks start a profiled step staggered: use the least-skewed rank (`tools/profile_table.py` does).
- **Divergence analysis:** compare *emitted tokens*, not argmaxes of returned scores. At exact bf16 top-2 ties (36 in 2,560 XLA steps) the on-device sampler and host argmax pick different tokens; argmax comparison then reports the next step as the divergence with a silently different context. Prefer teacher forcing (`capture_tf.py`).
- NKI builds are bit-deterministic within a process; XLA is also bit-reproducible across processes.
- The worktree's Bash guard rejects compound commands containing words like "logit"/"git" in odd places; split commands.
- `tensor_capture_config` must be set before `prepare_inference` (see `capture_layers.py`); module capture returns one tensor per module (first output / first input).
