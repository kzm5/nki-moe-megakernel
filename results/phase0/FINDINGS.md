# Phase 0 — Qwen3-30B-A3B on trn2: lab notebook

Machine: trn2.3xlarge (1 device, 4 logical cores at LNC=2, 96 GB HBM, 124 GiB host RAM + 64 GiB swap).
Model: Qwen3-30B-A3B, TP=4, LNC=2, bs=1. Branch: `phase0/trn2-setup`.
Venv: `~/venv-neuron-229` (Neuron SDK 2.29.0) unless stated. Ledger: [`ledger.jsonl`](ledger.jsonl) (copy of `~/runs/ledger.jsonl`).
Large artifacts (logs, reports, logits, profiles, compiled models) live in `~/runs/` and `~/qwen-30b-a3b/`, not in git.

## Key results so far

| Date | Config (trn2, seq 768, SDK 2.29) | Sampling | TKG p50 | TKG p99 | E2E tok/s | Validated | Source |
|---|---|---|---|---|---|---|---|
| 2026-11-11 | XLA | host, top-k 20 | 14.93 ms | 15.29 ms | 36.25 | n/a (reference) | `~/runs/xla_trn2_229_s768.log` |
| 2026-11-11 | XLA + library MoE TKG | host, top-k 20 | 10.39 ms | 11.00 ms | 43.35 | no | `~/runs/xlamoe_trn2_229_s768.log` |
| 2026-11-11 | Megakernel | on-device, top-k 20 | 10.41 ms | 10.77 ms | 88.57 | no | `~/runs/nki_trn2_229_s768.log` |
| 2026-11-11 | **XLA, greedy (fair)** | on-device greedy, logits out | 14.91 ms | 15.28 ms | 49.67 | reference | `~/runs/xla_greedy_s768.log` |
| 2026-11-11 | **XLA + library MoE, greedy (fair)** | on-device greedy, logits out | 10.30 ms | 10.98 ms | 63.62 | **no** (0/20 logit_validation) | `~/runs/xlamoe_greedy_s768.log` |
| 2026-11-11 | **Megakernel, greedy (fair)** | on-device greedy, logits out | 10.12 ms | 10.57 ms | 64.64 | **no** (0/20 logit_validation) | `~/runs/nki_greedy_s768.log` |
| 2026-11-11 | Library MoE vs megakernel, greedy, **n=5** | on-device greedy, logits out | 10.328 ± 0.036 vs **10.113 ± 0.020** ms | 11.09 vs **10.59** ms | – | no | Task 5 |
| 2026-11-11 | Library MoE vs megakernel, greedy, **seq 2048** | on-device greedy, logits out | 10.839 vs **10.233** ms (1.059×) | 11.36 vs **10.48** ms | – | no | Task 6 |
| 2026-11-11 | **Config D** (lib MoE + NxDI NKI attention), greedy, n=3 | on-device greedy, logits out | 12.076 ± 0.044 (768), 12.081 ± 0.101 (2048) | 12.46–12.63 / 12.47–12.75 | – | no | Day 2 Task 4 |
| 2026-11-12 | Library MoE vs megakernel, greedy, **seq 4096** | on-device greedy, logits out | 11.195 vs **10.441** ms (1.072×) | 11.91 vs **11.05** ms | – | no | Day 2 Task 7 |
| 2026-11-12 | XLA / lib MoE / megakernel, greedy, **n=5** (day 2) | on-device greedy, logits out | 15.076 ± 0.093 / 10.310 ± 0.008 / **10.154 ± 0.043** | 15.72 / 11.05 / **10.66** | – | ρ vs fp32 golden: 1 / 0.96 / 1.05 | Day 2 Task 6, Task 2 |
| 2026-11-12 | **Teacher-forced census** lib MoE / megakernel, 2,816 positions each | teacher-forced (XLA tokens) | – | – | – | **0 / 0 gross**; ρ 0.909 / 0.935; max KL 0.56 / 0.86; in-process deterministic | Day 3 T1 |

First three rows are an unfair comparison: the megakernel forces on-device sampling, the XLA configs sample on the host from full-vocab logits. Fair greedy rebuilds are Task 1–2 below.

## Run-procedure decisions (overnight, unattended)

- **Pushing enabled.** At 03:07 UTC `gh` was not on PATH and there were no GitHub credentials, so the seed commit was local only. The user then installed `gh` (logged in as `kzm5`) and created the fork. Remote `mine` = `https://github.com/kzm5/nki-moe-megakernel.git` (git auth via `gh auth setup-git`); `phase0/trn2-setup` is pushed after every task.
- One device job at a time; free disk checked before every compile.

---

## 2026-11-10/11 — Setup findings (seed)

### trn3 hardcoding blocked every trn2 megakernel load — Measured
- `main.py` set `NEURON_PLATFORM_TARGET_OVERRIDE = "trn3"` unconditionally in `main()`. Now set from `--platform-target` (default trn2), commit `9d5ed9a`.
- `megakernels/qwen3_moe/qwen_with_megakernel.py:83` (and `qwen_with_megakernel_blockwise.py:89`) set it to `trn3` at import, after `main()` had set it, so every megakernel build was compiled for NeuronCore v4. Load failed with `NEFF arch: v4, instance arch: v3` / `Could not load the model status=2 message=Invalid` (`~/runs/nki_trn2_229_s768.trn3_loadfail.log`). Removed in `3a22a56`.
- Still hardcoded (not changed): `NEURON_LOGICAL_NC_CONFIG = "2"` in both files (matches LNC=2; would conflict with `--logical-nc-config 1`); gpt_oss modules use `setdefault(..., "trn3")` and `NEURON_LOGICAL_NC_CONFIG = "1"`.

### Import-chain coupling to the Qwen3.6 code — Measured
- `nki_kernels/__init__.py` eagerly imported all Qwen3.6 kernels (DeltaNet, GQA, MoE verify, LM head). `deltanet/vendored/qkv_tkg.py:60` imports `DtypeMode`, which the nkilib bundled with neuronx-cc 2.25 lacks, so `import megakernels.qwen3_moe.qwen_with_megakernel` failed even though Qwen3 never uses those kernels. Made lazy via module `__getattr__` in `271cf6f`.

### Which kernel actually runs — Measured
- `--enable-nki` uses `megakernels/qwen3_moe/transformer_qwen3_moe_speculative.py` (imported by `qwen_with_megakernel.py:74`), which calls the vendored legacy `nki_kernels/_legacy/attention/attention_block_tkg.py` → `attention_tkg.py` and `_legacy/moe/moe_tkg`. `transformer_qwen.py` / `attn_fused_nki.py` / `moe_fused_nki.py` are not on this path.

### nkilib version requirements — Measured
| nkilib source | Result for Qwen3 megakernel |
|---|---|
| bundled with neuronx-cc 2.25 (SDK 2.30) | compile fails: `failed to resolve name 'dims.num_total_128_tiles_per_I'` (`_legacy/moe/selective_expert_impl.py:236`); field absent from `MLPTKGConstantsDimensionSizes` |
| fork `2.30_release` (`26628c5`) | no `DtypeMode` in `common_types.py`; not tested end to end |
| fork `2.29_release` (`ddbca35`) + SDK 2.29 (neuronx-cc 2.24.5133.0) | **works** (compiles, loads, benchmarks) |
- The editable `nki_library` install exposes `nkilib_src.nkilib`; the neuronx-cc-bundled `nkilib` redirects to it unless `NKILIB_FORCE_BUNDLED_LIBRARY` is set. Verified `nkilib.__file__` → `nki_kernels/nki-library/src/nkilib_src/nkilib/`.
- Submodule pinned to `2.29_release` (`22dc8cb`), `.gitmodules` branch + HTTPS URL (`36cbdc3`).

### seq_len % 256 at LNC=2 — Measured
- `_legacy/attention/attention_tkg.py:831` requires the per-core prior length `curr_sprior // sprior_n_prgs` to be a multiple of 128. At LNC=2 with bs=1 the prior is sharded across both cores, so `seq_len` must be a multiple of 256. `--seq-len 640` fails with `[NCC_INKI016] Sharded s_prior must be divisible by p_max. Got sharded s_prior=320` (`~/runs/nki_trn2_compile.log`). 768 passes. The speculative kernel's own docstring states the same rule.
- Other constraints on the active path (all satisfied at bs=1, S_tkg=1): B·S_tkg ≤ 128, d_head = 128, 8 Q / 1 KV heads per TP rank, MoE T ≤ 128, B=1 assumed by the mask builder.

### Unpushed submodule commit — Measured
- The superproject's recorded submodule pointer before this branch was `72ab554`, which does not exist in the `KevGomes1403/nki-library` remote (`git log 72ab554` → unknown revision). A fresh clone of `main` cannot check out the submodule. This branch repins to `ddbca35`.

### Host RAM limits — Measured
- Peak RSS during load/shard: XLA 118.3 GiB (seq 640), 118.9 GiB (768, swapped 37 GiB); library MoE 121.5 GiB; megakernel 121.5 GiB. Host has 124 GiB. Sharding-on-load with swap took 755 s vs 148 s without.
- 64 GiB `/swapfile` added (in `/etc/fstab`). Never run two device jobs at once.
- Hypothesis: `--save-sharded-checkpoint` removes the load-time sharding peak (NxDI loads `weights/tp{rank}_sharded_checkpoint.safetensors` directly, `application_base.py:389`).

### Environment
- Neuron apt: runtime-lib 2.32.31.0, collectives 2.32.28.0, dkms 2.28.0.0 (built for kernel 6.17.0-1007-aws). SDK 2.29 Python stack runs on this runtime.
- `~/venv-neuron-230` (SDK 2.30) is kept but unused: its bundled nkilib breaks the legacy MoE kernels.

### Three-way result (unfair) — Measured
| seq 768, SDK 2.29 | XLA | XLA + library MoE | Megakernel |
|---|---|---|---|
| TKG p50 / p99 | 14.93 / 15.29 ms | 10.39 / 11.00 ms | 10.41 / 10.77 ms |
| E2E p50 | 21,161 ms | 17,725 ms | 8,657 ms |
| Sampling | host top-k 20 | host top-k 20 | on-device top-k 20 (forced) |
- Library MoE over XLA: 1.44× TKG. Megakernel over library MoE: 1.00× TKG (trn3 README: 1.16×).
- Unfair: the megakernel forces on-device sampling (`qwen_with_megakernel.py:120,553`); the XLA configs ship full-vocab logits to the host every step (≈9 s/run extra). E2E numbers are not comparable.
- The megakernel returns token IDs only (`output_logits` off), so it cannot be validated as built.

### Open questions
- Where does the megakernel's trn3 advantage over library MoE go on trn2? (Task 4.)
- Does `transformer_qwen.py` (unused) still build on trn2?

---

## 2026-11-11 — Task 1: fair greedy rebuilds (seq 768)

Common flags: `--seq-len 768 --on-device-sampling --top-k 1 --do-sample "" --output-logits --save-sharded-checkpoint` (runner `~/runs/run_job.sh`, which adds `--model qwen3_moe --mode evaluate_single --platform-target trn2 --model-path ~/qwen-30b-a3b/hf_model --tp-degree 4 --logical-nc-config 2`).

```bash
bash ~/runs/run_job.sh xla_greedy_s768    ~/qwen-30b-a3b/traced_xla_trn2_229_s768_greedy    <common> [--skip-sharding]
bash ~/runs/run_job.sh xlamoe_greedy_s768 ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_greedy --moe-fused-nki-kernel-enabled <common>
bash ~/runs/run_job.sh nki_greedy_s768    ~/qwen-30b-a3b/traced_nki_trn2_229_s768_greedy    --enable-nki <common>
```

### `--top-k 1` alone is not greedy — Measured
- First XLA build used `--on-device-sampling --top-k 1` (do_sample left at main.py's default True). Saved logits vs emitted tokens: **227 / 2560 tokens (8.9%) were not the argmax**, e.g. step 79 of prompt 0: argmax ` legacy` (23.75) vs emitted ` value` (22.5).
- NxDI's sampler takes the multinomial path whenever `do_sample` is True (`modules/generation/sampling.py:459`), regardless of top_k. Hypothesis: with `global_topk=1` the distributed top-k feeds per-rank candidates into the multinomial, so a rank-local best can win.
- `main.py`'s `--do-sample` is `type=bool`, so only an empty string yields False: `--do-sample ""`. Rebuilt with it: 27 / 2560 mismatches remain, **all exact bf16 ties** (gap 0.0) → sampler is greedy. Config shows `do_sample: false, top_k: 1`.
- The non-greedy run is kept as `~/runs/xla_greedy_s768_dosample_true.log` (ledger `xla_greedy_s768_20261111`, status `superseded`); its captured logits are in `~/runs/val/dosample_true/`.

### Pre-sharded checkpoints — Measured
- Each rank's shard is 15.3 GB → ~57–61 GB per model (`weights/tp{0..3}_sharded_checkpoint.safetensors`). Pre-sharding at compile took 1380 s (swap-heavy, peak RSS 120 GiB).
- Loading pre-sharded weights: whole capture process (load + 20×128 tokens) 90 s, peak RSS 64 GiB.
- Recompiling into the same dir with `--skip-sharding` reuses existing shards (weights don't depend on sampler config): 23 min total, peak RSS 74 GiB.
- Decision: three pre-sharded builds (~180 GB) don't fit in 157 GB free. Ran Task 3's Process A (XLA expected-logit capture) right after the XLA build, then deleted the XLA build's `weights/` (57 GB; logged in `~/runs/deletions.log`) so library MoE and megakernel can both be pre-sharded for Tasks 4–5.

### E2E overhead with `--output-logits` — Hypothesis
- XLA greedy E2E 15.6 s vs CTE + 760 × TKG ≈ 11.7 s. The remaining ~4 s/run (was ~9 s with host sampling) is likely the per-step full-vocab logit gather + host copy that `--output-logits` adds. It applies to all three greedy builds equally.

### Build verification — Measured
| Build | Path | do_sample / top_k | output_logits | Kernel check | NEFF arch errors | Peak RSS | Wall |
|---|---|---|---|---|---|---|---|
| XLA | `traced_xla_trn2_229_s768_greedy` | False / 1 | True | n/a | 0 | 74.3 GiB (shards reused) | 23:01 |
| XLA + library MoE | `traced_xlamoe_trn2_229_s768_greedy` | False / 1 | True | 192× `Running MoE Fused NKI kernel` (48 layers × 4 TKG buckets), 0× `Conditions not met` | 0 | 117.4 GiB | 56:24 |
| Megakernel | `traced_nki_trn2_229_s768_greedy` | False / 1 | True | `--enable-nki` | 0 | 111.1 GiB | 49:37 |

Logs: `~/runs/{xla,xlamoe,nki}_greedy_s768.log`; reports: `~/runs/{xla,xlamoe,nki}_greedy_s768/benchmark_report.json`.

## 2026-11-11 — Task 2: fair three-way benchmark (seq 768, greedy on-device, logits returned)

Same runs as Task 1 (`evaluate_single` benchmarks after compile+load).

| | XLA | XLA + library MoE | Megakernel |
|---|---|---|---|
| TKG p50 | 14.91 ms | 10.30 ms | **10.12 ms** |
| TKG p99 | 15.28 ms | 10.98 ms | **10.57 ms** |
| TKG throughput | 67.2 tok/s | 96.6 tok/s | 98.8 tok/s |
| CTE p50 | 380.0 ms | 382.0 ms | 401.6 ms |
| E2E p50 | 15,477 ms | 12,058 ms | 11,876 ms |
| E2E throughput | 49.7 tok/s | 63.6 tok/s | 64.6 tok/s |

| Split | TKG p50 | TKG p99 | E2E p50 |
|---|---|---|---|
| Library MoE over XLA | **1.45×** | 1.39× | 1.28× |
| Megakernel over library MoE | **1.018×** | 1.039× | 1.015× |
| Megakernel over XLA | 1.47× | 1.45× | 1.30× |

What it shows (Measured): on trn2 the library MoE TKG kernel delivers almost all of the decode gain over XLA; the megakernel adds ~1.8% at p50 (~0.18 ms/step). With sampling and logit return made identical, the earlier 2.44× E2E gap collapses to 1.015× — it was the forced on-device sampling, not fusion.
Surprising: trn3 README shows megakernel 1.16× over library MoE; trn2 shows 1.02×. Whether 1.8% exceeds run-to-run noise is Task 5.
Megakernel CTE is 5% slower (401.6 vs 382 ms) even though CTE is XLA in both — Hypothesis: different CTE graph flags from the megakernel model class (`get_compiler_args`), not the kernel.
Open: E2E still ≈ CTE + 760×TKG + ~3.6 s; Hypothesis: per-step full-vocab logit gather/copy for `--output-logits`.

## 2026-11-11 — Task 3: correctness check (logits A/B, never two models in RAM)

```bash
# Process A (reference): XLA greedy build, 20 fixed prompts (validation_prompts.json) x 128 new tokens
python ~/runs/val/capture.py ~/qwen-30b-a3b/traced_xla_trn2_229_s768_greedy          # -> ~/runs/val/expected_p{00..19}.pt (fp32, 1.5 GB)
# Process B: logit_validation (same call as main.py check_accuracy_logits), two tol maps
python ~/runs/val/validate.py libmoe ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_greedy
python ~/runs/val/validate.py nki    ~/qwen-30b-a3b/traced_nki_trn2_229_s768_greedy
python ~/runs/val/analyze.py libmoe nki                                               # -> validation_summary.json
```
Scripts: [`scripts/`](scripts/). Repo tol_map `{None:(1e-5,0.05), 1000/50/5:(1e-5,0.03)}`; "rtol 1e-2" = same keys with rtol 0.01. `divergence_difference_tol=0.001`. Per-position arrays (max-abs diff, top-1, KL) are in `~/runs/val/results_{libmoe,nki}.json`; first-pass actual logits in `~/runs/val/actual_*.pt`.

Reference sanity: XLA greedy logits' argmax equals the emitted token at 2533/2560 positions; all 27 others are exact bf16 ties.

| vs XLA greedy reference | Library MoE | Megakernel |
|---|---|---|
| logit_validation pass, repo tol_map | **0 / 20** | **0 / 20** |
| logit_validation pass, rtol 1e-2 | **0 / 20** | **0 / 20** |
| Prompts with no top-1 divergence in 128 tokens | 1 / 20 | 2 / 20 |
| Median first-divergence index | 31 | 26 |
| Median pre-divergence mean KL (nats) | 1.8e-3 | 1.9e-3 |
| Divergences at a near-tie (expected top1−top2 gap ≤ 0.375) | 15 / 19 | 11 / 18 |

Per-prompt first divergence (`first_div`, −1 = none), pre-divergence max-abs logit diff, and expected top-1/top-2 gap at the divergence:

| p | libMoE first_div | libMoE pre max-abs | libMoE gap@div | nki first_div | nki pre max-abs | nki gap@div |
|---|---|---|---|---|---|---|
| 0 | 31 | 2.78 | 0.0 | 32 | 2.88 | 0.125 |
| 1 | 15 | 1.00 | 8.75 | 15 | 1.41 | 8.75 |
| 2 | 111 | 3.36 | 3.125 | 5 | 0.31 | 2.25 |
| 3 | 27 | 1.13 | 0.125 | 27 | 1.75 | 0.125 |
| 4 | −1 | 2.50 | – | −1 | 3.28 | – |
| 5 | 112 | 6.06 | 8.0 | 2 | 0.36 | 2.625 |
| 6 | 4 | 0.69 | 0.375 | 4 | 0.69 | 0.375 |
| 7 | 45 | 5.75 | 5.0 | 45 | 5.84 | 5.0 |
| 8 | 77 | 1.31 | 0.0 | 5 | 0.53 | 0.5 |
| 9 | 31 | 0.81 | 0.0 | 2 | 0.41 | 0.0 |
| 10 | 20 | 0.81 | 0.25 | 21 | 1.09 | 3.125 |
| 11 | 71 | 3.27 | 0.125 | 102 | 12.37 | 0.25 |
| 12 | 70 | 1.38 | 0.375 | 70 | 1.53 | 0.375 |
| 13 | 80 | 4.00 | 0.0 | 14 | 1.84 | 1.375 |
| 14 | 42 | 1.25 | 0.25 | 42 | 1.00 | 0.25 |
| 15 | 5 | 0.39 | 0.0 | 5 | 0.47 | 0.0 |
| 16 | 26 | 1.81 | 0.0 | 26 | 1.31 | 0.0 |
| 17 | 27 | 3.00 | 0.375 | 26 | 1.56 | 0.125 |
| 18 | 2 | 0.35 | 0.0 | 12 | 0.70 | 0.0 |
| 19 | 12 | 1.53 | 0.0 | −1 | 4.75 | – |

What it shows:
- **Measured:** neither NKI build passes NxDI `logit_validation` against the XLA greedy reference at either tolerance. Validated = false for both (ledger updated).
- **Measured:** the two NKI builds are statistically alike vs XLA (same pre-divergence KL ~2e-3 nats, similar divergence positions; 9 prompts diverge at the identical index). The megakernel is not measurably worse than the library MoE kernel.
- **Measured:** even without any token divergence (p04) the failure is logit magnitude, not argmax: per-position error reaches 3.5–7.7% of max |logit| (e.g. token 36: top-k None error 0.077 > 0.05), while KL stays ~7e-4.
- **Hypothesis:** most divergences are bf16 near-ties (gap ≤ 0.375 ≈ 3 bf16 ulps at |logit|≈20), i.e. numerically benign; the repo tol_map is tighter than bf16 MoE accumulation-order differences allow.
- **Surprising:** p01@15 (gap 8.75) and p06@4 diverge at the same index and pick the *same* off-ranked token (rank 329 / 833 in the reference) in both NKI builds, which use different MoE kernels. Hypothesis: the XLA reference is the outlier there (e.g. a router top-8 near-tie resolved differently by XLA's torch blockwise path), not an NKI bug.
- Note: whole-sequence top-1/KL/max-abs in `results_*.json` are free-running after the first divergence and are not meaningful past it; use the pre-divergence columns.

Open questions:
- Is the XLA reference itself deterministic run-to-run (XLA vs XLA)? That bounds the achievable tolerance.
- Do p01/p06 trace back to a router near-tie? Compare per-layer router logits at those positions.
- Teacher-forced comparison (feed reference tokens every step) would give per-position metrics without divergence cascades.

## 2026-11-11 — Task 4: per-layer profile, library MoE vs megakernel — FAILED (blocked)

```bash
python ~/runs/prof/profile_tkg.py libmoe ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_greedy ~/runs/prof/libmoe
# sets NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_DEVICE_PROFILE=1 NEURON_RT_INSPECT_OUTPUT_DIR=..., then prefill + 4 greedy decode steps
```

| Attempt | Result |
|---|---|
| 1. runtime inspect capture (library MoE) | Captured 9 NEFFs × 4 ranks; 4 sets of 1.8–2.4 GB NTFFs (non-decode graphs) filled the disk → `NRT:nrt_profile_serialize Failed to serialize a profile`, exit 1 |
| 2. read the NTFFs (`neuron-profile show-session`, `neuron-explorer view`) | `NTFF version 7 is not supported in this version of the tool (supported: 1 - 6). Please upgrade aws-neuronx-tools.` |

- **Measured:** runtime 2.32.31 (installed for SDK 2.30) writes NTFF v7; installed `aws-neuronx-tools` is 2.28.23 (supports v1–6). Upgrading tools is an apt change, out of scope tonight. Megakernel capture skipped (same blocker, same disk risk).
- Kept: small NTFFs + all NEFFs in `~/runs/prof/libmoe/` (321 MB) for re-analysis after a tools upgrade; deleted the 4 large sets (logged in `~/runs/deletions.log`).
- Unblock: `sudo apt-get install aws-neuronx-tools=2.30.10.0*` (SDK 2.30 tools) — then rerun with `NEURON_RT_INSPECT_*` restricted to the TKG NEFF (or `neuron-profile capture -n <tkg.neff>` with `--collectives-workers-per-node 4`) to avoid profiling the multi-GB CTE/weight-layout graphs.

Candidate explanations for megakernel ≈ library MoE on trn2 (1.02×) vs 1.16× on trn3 — **all Hypothesis, none measured**:
| Candidate | Status | Reasoning |
|---|---|---|
| SBUF spill / lost double-buffering on trn2's smaller SBUF | Hypothesis (plausible) | The speculative megakernel sizes its allocator at `SBM_SIZE_BYTES = 200 KiB` per partition and double-buffers per-layer weights (`interleave_degree=2`); trn2 SBUF is 192 KiB/partition vs 256 KiB+ on trn3 (unverified spec), so prefetch may serialize. Needs `spill_reload_bytes` and DMA/compute overlap from the profile. |
| Lower graph-boundary cost on trn2 | Hypothesis | The megakernel's advantage is removing per-layer HBM round-trips/graph boundaries; if trn2's XLA graph already overlaps those well, there's less to remove. |
| AllReduce overhead | Hypothesis | Both configs do 2 AllReduces per layer; the megakernel's SB2SB AR-gather may be relatively costlier on trn2's collectives. |
| Something else | Hypothesis | e.g. the vendored legacy kernels were tuned for trn3 tile shapes (`get_compiler_args` lnc/tiling flags). |

## 2026-11-11 — Task 5: noise floor (5 reruns each, pre-sharded, `--skip-compile`)

```bash
bash ~/runs/noise.sh   # alternates: libMoE r1, nki r1, libMoE r2, ... (each a fresh process: load + evaluate_single benchmark)
```
Each rerun: ~9.5 min wall (pre-sharded load), peak RSS 54–65 GiB. Logs `~/runs/noise_{libmoe,nki}_r{1..5}.log`; stats `~/runs/noise_summary.json`.

| TKG p50 (ms) | r1 | r2 | r3 | r4 | r5 | mean | std | min | max |
|---|---|---|---|---|---|---|---|---|---|
| Library MoE | 10.341 | 10.305 | 10.386 | 10.304 | 10.304 | **10.328** | 0.036 | 10.304 | 10.386 |
| Megakernel | 10.079 | 10.128 | 10.110 | 10.121 | 10.127 | **10.113** | 0.020 | 10.079 | 10.128 |

| | Library MoE | Megakernel |
|---|---|---|
| TKG p99 mean ± std | 11.092 ± 0.046 ms | 10.589 ± 0.059 ms |
| E2E p50 mean ± std | 11,982 ± 59 ms | 11,758 ± 53 ms |

What it shows (Measured): the megakernel is faster than library MoE by **0.215 ms/step (2.1%) at p50**, ~6–10× the run-to-run std (Welch t ≈ 11.6, ranges do not overlap: slowest megakernel 10.128 < fastest library MoE 10.304). At p99 the gap is 0.50 ms (4.5%). The Task 2 single-run numbers (10.30 / 10.12) sit inside these distributions.
So the trn2 fusion gain is real but small: 1.021× p50, 1.047× p99, vs 1.16× on trn3 (README).
Open: is the p99 gap (tail) from fewer graph boundaries, i.e. less jitter, rather than a faster steady state? Needs the per-layer profile (Task 4, blocked).

## 2026-11-11 — Task 6: sequence-length sweep (greedy flags, no pre-shard)

```bash
bash ~/runs/sweep.sh   # for s in 256 512 1024 2048: run_job.sh sweep_{libmoe,nki}_s$s ... --seq-len $s --on-device-sampling --top-k 1 --do-sample "" --output-logits
```
All 8 builds compiled, loaded and benchmarked (exit 0, no NEFF arch errors); 17–45 min each, peak RSS 119.5–122.1 GiB. Logs `~/runs/sweep_{libmoe,nki}_s{256,512,1024,2048}.log`, status `~/runs/sweep_status.log`. 768 row = Task 5 means (n=5); others single runs (Task 5 std ≈ 0.02–0.04 ms at p50).

| seq_len | libMoE TKG p50 | Megakernel TKG p50 | Speedup p50 | libMoE TKG p99 | Megakernel TKG p99 | Speedup p99 |
|---|---|---|---|---|---|---|
| 256 | 10.206 | 10.226 | 0.998× | 10.533 | 10.564 | 0.997× |
| 512 | 10.379 | 10.194 | 1.018× | 10.752 | 10.808 | 0.995× |
| 768 | 10.328 | 10.113 | 1.021× | 11.092 | 10.589 | 1.047× |
| 1024 | 10.655 | 10.227 | 1.042× | 11.082 | 10.709 | 1.035× |
| 2048 | 10.839 | 10.233 | **1.059×** | 11.360 | 10.482 | **1.084×** |

CTE p50: library MoE 382.0–382.4 ms, megakernel 401.6–401.9 ms at every length.

What it shows:
- **Measured:** megakernel decode latency is flat across context (10.11–10.23 ms from 256 to 2048); library MoE decode grows 10.21 → 10.84 ms (+6%). The fusion gain is zero at 256 and grows with context to 1.06× (p50) / 1.08× (p99) at 2048.
- **Hypothesis:** the growth is in the XLA attention path of the library-MoE config (KV read/QK/softmax scale with context and pay per-layer graph-boundary/HBM round-trips), while the megakernel's fused attention (`attention_block_tkg`, s_prior sharded across both LNC cores) hides the extra KV traffic. The MoE half is the same work in both. Confirming needs the per-layer profile (Task 4).
- **Surprising:** the 768-token gain (2.1%) understates the megakernel; Qwen3.6-style long-context decode is where it pays. The README's trn3 number (1.16×) was at 640 tokens; on trn2 even 2048 gives only 1.06×.
- **Measured:** the megakernel's CTE is a constant +19.5 ms (+5.1%) slower — CTE is XLA in both, so this is the megakernel model class's CTE compile flags or graph, not the kernel.

Open questions:
- Does the megakernel stay flat at 4096/8192 (attention_tkg switches to flash-attention tiling above 8K per core)?
- Repeat 256/512 with n=5: the 256 and 512 differences are within ~1–5× noise.

## 2026-11-11 — Stale items in `docs/RESEARCH_PLAN.md` (rev 16 export)

Noted here for the next re-export from the master doc; the snapshot itself is unchanged.

| Plan location | Says | Current state (Measured unless noted) |
|---|---|---|
| "Phase 0 results so far (Nov 10)" heading | Nov 10 | Numbers are from the Nov 11 fair rebuild (Tasks 1–6) |
| Repo finding 1 | `main.py` forces trn3 | Fixed: `9d5ed9a` (`main.py`), `3a22a56` (`qwen_with_megakernel*.py`) |
| Repo finding 4 | submodule pins `2.30_release` | Pinned to `2.29_release` (`ddbca35`), `22dc8cb`/`36cbdc3` |
| Repo finding 5 | submodule URL is SSH | HTTPS since `36cbdc3` |
| Repo finding 7 | trn3 skill is repo default | Flipped 2026-11-11: `nki-kernel-optimizer` (trn2) is default; CLAUDE.md hardware now trn2 |
| Process findings | "35 ledger lines" | Overnight ledger has 33 lines |
| Phase 0 item 2 | 10 runs per config | Overnight did 5 each for library MoE and megakernel, none for XLA (10× all three is queued 2026-11-12) |
| Phase 0 item 3 | audit inherited skills | Not done |
| Phase 0 exit | within 5% of Spring numbers on same hardware | Not applicable: Spring numbers are trn3, these are trn2 |
| Risks: SDK drift row | reproduce on 2.27 first | 2.27 never tried; reproduction is on SDK 2.29 (only working combo) |
| Open question | will Gomes push `72ab554` | Moot for this branch (repinned to `ddbca35`); still breaks a fresh clone of upstream `main` |
| What-it-took item 1 | `DtypeMode` only from 2.31 | Measured absent in bundled 2.25 nkilib and fork `2.30_release`; presence in 2.31 not tested (Hypothesis) |

## 2026-11-11 — Task 1 (day 2): per-layer decode profile, library MoE vs megakernel, seq 768 and 2048

```bash
sudo apt-mark unhold aws-neuronx-tools && sudo apt-get install aws-neuronx-tools=2.30.10.0* && sudo apt-mark hold aws-neuronx-tools
bash ~/runs/prof/run_prof.sh prof_{libmoe,nki}_s{768,2048} {libmoe,nki} <build> {768,2048}   # profile_decode.py
python ~/runs/prof/neffdump.py <build>/model.pt <out>/neffs                                     # TKG bucket NEFFs
python tools/profile_table.py OUT.json LABEL=NEFF:NTFF_GLOB ...                                 # summary-json -> per layer
python tools/profile_breakdown.py PROF.json ...                                                 # per-instruction json
```
Builds: `traced_xlamoe_trn2_229_s768_greedy`, `traced_nki_trn2_229_s768_greedy`, `traced_{libmoe,nki}_trn2_229_s2048_greedy_sweep` (weights shard on load).
Scripts: [`scripts/`](scripts/); tables `~/runs/prof/tables/*.json` (copied to [`profile/`](profile/)).

### Capture method — Measured
| Attempt | Result |
|---|---|
| 1. `torch.classes.neuron.Runtime().profile_enable` around one TKG call | No files: NxDI's SPMD execution path isn't covered |
| 2. ctypes `nrt_inspect_begin/stop` with `NEURON_RT_INSPECT_DEVICE_PROFILE=session` | 4 × 123 MB NTFF readable by tools 2.30, but no NEFF → no `view` |
| 3. Same, mode `model`, two windows per process: TKG call 6 (bucket 128) and the call at position `seq_len−4` (largest bucket) | Works: ~1 GB per capture (8 NTFF), NEFFs from `SPMDModel.save_neff` (`neffdump.py`); mismatched NEFF/NTFF is rejected by `view` |
- Env vars set in-process after load and before `nrt_inspect_begin()` keep CTE out of the capture (yesterday's whole-process inspect wrote 1.8–2.4 GB per NEFF and filled the disk).
- **Rank skew (Measured):** the four ranks enter a profiled step staggered; the early ones idle inside collectives (graph time per rank e.g. 21.9 / 9.3 / 65.7 / 88.9 ms, CC 13.6 / 0.9 / 57.4 / 80.6 ms) while engine and DMA times are identical across ranks. Tables use the least-skewed rank (minimum `total_time`, 9.2–9.9 ms, vs measured TKG p50 10.1–10.8 ms incl. host overhead).
- Per-layer = whole TKG graph / 48 (includes embedding, LM head, sampling — same convention assumed for Gomes's Table 5). n = 1 step per bucket.

### Table 5 on trn2 vs Gomes's trn3 (µs per layer unless noted)
| Metric | trn3 lib MoE (Gomes, 640 tok) | trn3 megakernel | trn2 lib MoE, s768 b768 | trn2 megakernel, s768 b768 | trn2 lib MoE, s2048 b2048 | trn2 megakernel, s2048 b2048 |
|---|---|---|---|---|---|---|
| Wall time / layer | 165.26 | 134.47 | 203.6 | **192.8** | 205.4 | **193.0** |
| Sync engine | 53.52 | 19.05 | 60.1 | 58.5 | 59.1 | 58.4 |
| DMA active | 91.99 | 66.50 | 128.3 | 123.5 | 128.5 | 125.7 |
| Tensor engine | 48.86 | 48.15 | 62.8 | 58.4 | 63.9 | 59.3 |
| CC ops (AllReduce) | 13.45 | 27.77 | 20.0 | 28.8 | 19.6 | 28.8 |
| DMA packets / layer | – | −43% vs lib MoE | 42,740 | 43,748 (+2%) | 43,135 | 43,865 (+2%) |
| Avg DMA packet bytes | – | 2.7× lib MoE | 787 | 783 (1.0×) | 795 | 796 (1.0×) |
| Scalar / Vector / GpSimd | – | – | 55.0 / 47.6 / 64.1 | 29.8 / 49.3 / 56.4 | 55.0 / 49.5 / 64.7 | 30.1 / 49.4 / 56.6 |
| HBM read MB / layer | – | – | 33.4 | 33.6 | 34.0 | 34.3 |
| Spill reload / save (bytes per graph) | – | – | 212,864 / 122,888 | 19,328 / 36,872 | 212,864 / 122,888 | 24,448 / 36,872 |
| Graph time (ms) | – | – | 9.77 | 9.26 | 9.86 | 9.27 |

Smallest bucket (128) for comparison: lib MoE 193.5 (s768) / 193.6 (s2048) µs/layer; megakernel 192.5 / 192.9.

### Where the trn2 gain comes from
| Explanation | Verdict | Evidence |
|---|---|---|
| Attention scaling with context | **Supported (Measured; op identity Hypothesis)** | Lib MoE wall grows +10.1 µs/layer (128→768) and +11.8 (128→2048); megakernel +0.3 / +0.1. At bucket 128 the two are equal (193.5 vs 192.5). Graph-time gap at the largest bucket (0.51 ms at 768, 0.59 ms at 2048) matches the benchmark gap at 2048 (0.61 ms). Per-instruction breakdown (lib MoE s2048): NKI MoE kernel and `attention_base.py`-tagged ops don't grow; the growth (TensorMatrix +13.3, Vector +7.4, GpSimd +7.3, Tensor +6.7 µs/layer summed duration) is in untagged XLA ops whose extra HBM read (+0.93 MB/layer) equals the extra K+V bytes (≈0.98 MB/layer/rank). Hypothesis: those are XLA's QK/softmax/PV over the bucket length. |
| SBUF spill / lost double-buffering | **Not supported (Measured)** | Megakernel spill reload 19–24 KB per graph (lib MoE 213 KB); its DMA-active and tensor time are slightly below lib MoE, not above. |
| Graph-boundary cost | **Not supported on trn2 (Measured)** | Sync engine 58.5 vs 60.1 µs/layer (trn3: 19.1 vs 53.5); DMA packets +2% and same average size (trn3: −43%, 2.7× larger). The trn3 boundary savings don't appear here. Why: Hypothesis (trn2 lib-MoE graph already has few boundaries? megakernel's trn3-tuned DMA tiling?). |
| Collectives | **Offsetting cost (Measured)** | Megakernel CC +8.8–9.2 µs/layer over lib MoE (trn3: +14.3). Together with its scalar-engine saving (−25 µs summed) the short-context wall is a wash. |

What it shows: on trn2 the megakernel's advantage over library MoE is the context-dependent attention cost that XLA attention pays and the fused NKI attention doesn't. Fusion itself (sync/DMA boundary savings) contributes nothing measurable; extra collective time cancels its other savings. This is the confound the plan flagged: the trn2 "fusion gain" is an attention-kernel gain. Task 4 (config D) tests it directly.

Surprising: megakernel HBM reads and DMA packets are identical across buckets (all its bucket NEFFs are the same size, 5.17/5.22 MB): its attention reads the full `seq_len` KV every step, which is still cheap (≈1 MB/layer at 2048).

Open: identify the growing XLA ops (source tags missing); n=1 step per bucket (no variance); whether Gomes's trn3 numbers used the same per-layer convention.

## 2026-11-11 — Task 2 (day 2): fp32 golden reference

```bash
python tools/golden_fp32.py --check-hf "The capital of France is"                       # bf16 layerwise vs stock HF
python tools/golden_fp32.py --dtype fp32 --inputs "$HOME/runs/val/expected_p00.pt" --out ~/runs/golden_timing   # timing
python tools/golden_fp32.py --dtype fp32 --inputs "$HOME/runs/val/expected_p*.pt" --out ~/runs/golden            # 20 prompts
python tools/compare_golden.py                                                           # -> ~/runs/golden/compare.json
```
`tools/golden_fp32.py`: stock HF `Qwen3MoeDecoderLayer` (transformers 4.57.6, sdpa), one layer resident; each layer's weights are read once and all 20 prompts' hidden states pass through it before it is freed. Teacher-forced: one forward per prompt over prompt + XLA's 128 greedy tokens; saves the 128 logit rows that predict each generated token, plus per-layer router top-9 (`~/runs/golden/golden_pNN.pt`, 1.5 GB).

### Script validation — Measured
| Check | Result |
|---|---|
| bf16 layerwise vs stock HF `Qwen3MoeForCausalLM` (bf16), "The capital of France is" (5 tokens) | **bit-identical** logits (max-abs 0.0) |
| fp32, 1 prompt (134 tokens) | 62.3 s wall, 4.3 GiB peak RSS |
| fp32, 20 prompts (2,764 tokens) | 92 s compute, 106 s wall, **4.4 GiB peak RSS** |

### Builds vs golden, positions before each build's first divergence — Measured
| vs fp32 golden | XLA | Library MoE | Megakernel |
|---|---|---|---|
| Positions | 2,560 (all) | 936 | 711 |
| rel-L2 mean / median | 0.0417 / 0.0281 | 0.0418 / 0.0276 | 0.0460 / 0.0282 |
| max-abs (max / mean) | 10.81 / 0.70 | 10.07 / 0.67 | 12.08 / 0.78 |
| KL(golden‖build) mean | 1.89e-3 | 2.24e-3 | 2.24e-3 |
| top-1 agreement with golden | 98.6% | 98.9% | 99.0% |
| **ρ = err(build)/err(XLA), same positions** | 1 | **0.957** | **1.054** |
| ρ_KL | 1 | 1.06 | 1.22 |

Both NKI builds are within the ρ ≤ 1.25 strawman gate on these positions: before divergence they are as accurate as XLA against fp32.

### The divergence positions themselves — Measured

> **Correction (day 3):** the "identical context" premise below is false for all 16 "gross" cases — the emitted tokens had already split at an exact tie one or more steps earlier. See "Day 3: the day-2 gross single-step errors were an analysis artifact". The gross/benign split and the defect hypothesis in this section are retracted.
At a build's first divergence the context is still identical to XLA's, so those logits are comparable too (they are excluded from ρ above). Two classes separate cleanly:

| | Library MoE (19 divergences) | Megakernel (18) |
|---|---|---|
| Benign (KL(golden‖build) ≤ 0.02, golden top1−top2 ≤ 0.5) | 12 — golden picks the build's token in 7 | 9 — golden picks the build's token in 6 |
| **Gross (KL 0.9–25 nats, ρ at div 8–63)** | **7 — golden picks XLA's token in all 7** | **9 — golden picks XLA's token in all 9** |

Gross divergences (KL(golden‖XLA) at the same positions ≤ 0.01):
| Build | Prompt@pos | ρ at div | KL(golden‖build) | KL(golden‖XLA) | Golden top1−top2 |
|---|---|---|---|---|---|
| lib MoE | p03@27 | 36.8 | 24.78 | 0.0022 | 0.003 |
| lib MoE | p06@4 | 32.4 | 20.56 | 0.0093 | 0.157 |
| megakernel | p06@4 | 32.7 | 20.36 | 0.0093 | 0.157 |
| megakernel | p13@14 | 58.4 | 17.93 | 0.0034 | 1.178 |
| megakernel | p02@5 | 63.0 | 16.45 | 0.0002 | 2.197 |
| lib MoE | p02@111 | 40.8 | 15.78 | 0.0002 | 3.024 |
| lib MoE | p01@15 | 26.4 | 14.96 | 0.0000 | 8.625 |
| megakernel | p01@15 | 26.0 | 14.62 | 0.0000 | 8.625 |
| lib MoE | p05@112 | 28.8 | 14.04 | 0.0001 | 7.521 |
| megakernel | p05@2 | 33.1 | 9.16 | 0.0001 | 2.671 |
| megakernel | p08@5 | 11.8 | 7.40 | 0.0021 | 0.455 |
| lib MoE | p17@27 | 7.9 | 3.24 | 0.0014 | 0.471 |
| megakernel | p00@32 | 28.0 | 2.62 | 0.0012 | 0.140 |
| megakernel | p10@21 | 40.4 | 1.21 | 0.0000 | 3.146 |
| lib MoE | p07@45 | 13.4 | 1.02 | 0.0000 | 4.980 |
| megakernel | p07@45 | 13.2 | 0.88 | 0.0000 | 4.980 |

The two "same unlikely token" spots — **golden sides with XLA; the XLA-outlier hypothesis is refuted (Measured):**
| Spot | Context (tail) | Golden top-1 | XLA token | Both NKI builds' token (golden rank, value) | NKI top-3 |
|---|---|---|---|---|---|
| p01@15 | "…capital of the United Kingdom is London. The capital of Spain" | " is" (33.6) | " is" | " United" (rank 298, 13.7) | " United" 25.8 / " Netherlands" 23.3 / " USA" 23.1 |
| p06@4 | "…I want to go for a walk. I have a new" | " pair" (23.0) | " pair" | " of" (rank 880, 11.9) | " of" 31.3 / " to" 28.0 / " on" 22.1 |

What it shows:
- **Measured:** the benign divergences are bf16 near-ties, and there golden is split between XLA and the NKI build — neither is "right" at a near-tie.
- **Measured:** 7/19 (lib MoE) and 9/18 (megakernel) first divergences are not near-ties but single-step gross errors (KL up to 25 nats at a position whose preceding outputs matched golden to ~2e-3), always against golden. Three occur at the same position in both builds (p01@15, p06@4, p07@45) with near-identical values (" United" 25.75 vs 25.38).
- **Hypothesis:** a shared, input-triggered defect in both NKI paths. They share the NKI MoE TKG kernel family (the megakernel vendors the legacy `moe_tkg`). The NKI top-3 at the spots look like predictions for a different context (country names after "The capital of"; " of" as if " pair" were already present) — wrong expert outputs or stale hidden state, not rounding.
- **Gate note:** ρ over pre-divergence positions (the Task 2 metric) cannot see these by construction. The Task 5 trigger (ρ > 1.25) is not met as specified; ρ *at the divergence position* is 8–63 for the gross cases.

## 2026-11-11 — Task 3 (day 2): XLA self-consistency

```bash
bash ~/runs/xla_selfcons.sh   # preshard.py xla (505 s), then capture_to.py xla ... ~/runs/val/xla_self_{A,B} in two separate processes
python tools/compare_runs.py ~/runs/val/xla_selfcons.json ref=~/runs/val A=~/runs/val/xla_self_A B=~/runs/val/xla_self_B --golden ~/runs/golden
```
Same 20 prompts × 128 greedy tokens as Task 3 (day 1). `ref` = day-1 capture (`valA_xla_capture_20261111`).

| Pair | Prompts with identical tokens | Prompts with bit-identical outputs (all 128 × 151,936 values) |
|---|---|---|
| ref vs A | 20 / 20 | **20 / 20** |
| ref vs B | 20 / 20 | **20 / 20** |
| A vs B | 20 / 20 | **20 / 20** |

What it shows (Measured): the XLA greedy build is bit-deterministic across processes and days (same NEFFs, shards rebuilt by `preshard.py` vs at compile time). Run-to-run nondeterminism contributes nothing to the XLA reference; every NKI-vs-XLA difference is a real numerical difference of the NKI builds. Its error vs fp32 golden (rel-L2 0.0417) is the same in all three runs.

Notes: capture A 9.0 min wall (cold page cache, peak RSS 49 GiB), B 82 s (warm, 64 GiB). Open: is the megakernel equally deterministic? (not tested).

## 2026-11-11 — Task 4 (day 2): separate attention from fusion (config D)

NxDI 0.x in `~/venv-neuron-229` has its own NKI attention token-generation kernel that runs alongside `--moe-fused-nki-kernel-enabled`: `attn_block_tkg_nki_kernel_enabled` (`modules/attention/attention_base.py:1753`), which supports Qwen3's pre-RoPE Q/K RMSNorm. `megakernels/qwen3_moe/qwen.py:538` requires `--attn-block-tkg-nki-kernel-cascaded-attention` with it, and the NxDI GQA path requires `--fused-qkv` with `--qkv-kernel-enabled` (attempt 1 failed in 7 s: `AssertionError: QKV kernel only supported when fused_qkv is TRUE`, `gqa.py:497`).

```bash
bash ~/runs/configD.sh   # per seq_len: run_job.sh (compile+bench) then 2 x --skip-compile reruns
# D flags: --moe-fused-nki-kernel-enabled --attn-block-tkg-nki-kernel-enabled --attn-block-tkg-nki-kernel-cascaded-attention
#          --qkv-kernel-enabled --fused-qkv --on-device-sampling --top-k 1 --do-sample "" --output-logits --seq-len {768,2048}
```
Verified in the logs: `[attn-block-tkg]` SBUF-manager lines (17 at 768, 22 at 2048) and 191× `Running MoE Fused NKI kernel`; saved `neuron_config` has all four flags. Compile 824 s (768) / 1,161 s (2048); reruns 12.7–39 min wall (weights shard on load, peak RSS 121 GiB, I/O-bound).

### Four configs, TKG p50 (ms) — Measured
| Config | seq 768 | seq 2048 | Δ 768→2048 |
|---|---|---|---|
| A. XLA | 14.91 (n=1) | – | – |
| B. XLA + lib MoE | 10.328 ± 0.036 (n=5) | 10.839 (n=1) | +0.511 |
| **D. XLA + lib MoE + NxDI NKI attention** | **12.076 ± 0.044 (n=3)** | **12.081 ± 0.101 (n=3)** | **+0.005** |
| C. Megakernel | 10.113 ± 0.020 (n=5) | 10.233 (n=1) | +0.120 |

p99: D 12.46–12.63 (768), 12.47–12.75 (2048). E2E p50: D 19.0–19.3 s (768) vs B 12.0 s and C 11.8 s.
D runs: 12.113 / 12.087 / 12.027 (768); 12.197 / 12.029 / 12.016 (2048).

### Split of the megakernel's gain over lib MoE
| Component | 768 | 2048 | Basis |
|---|---|---|---|
| Total gain B→C | 0.215 ms | 0.606 ms | measured |
| Context-dependent part (B's slope − C's slope, from 768) | – | 0.391 ms | measured slopes |
| NKI attention removes the context slope? | – | yes: D and C both ≈ flat (+0.005, +0.12) vs B +0.51 | measured |
| Adding NxDI's NKI attention to B (B→D) | **+1.75 ms (slower)** | **+1.24 ms (slower)** | measured |
| C vs D ("fusion + Gomes's attention" vs NxDI attention) | −1.96 ms | −1.85 ms | measured |

What it shows:
- **Measured:** NKI attention on its own eliminates the context-length growth of decode latency (D is flat from 768 to 2048, like the megakernel; lib MoE's XLA attention grows 0.51 ms). This confirms Task 1's profile: the megakernel's context-dependent advantage is an attention-kernel effect.
- **Measured:** NxDI's attention-block TKG kernel is **1.17× slower than XLA attention at 768** on trn2 (+1.75 ms/step ≈ +36 µs/layer), and still slower at 2048. So the four-config design cannot cleanly split "attention kernel" from "fusion": the only available NKI attention stand-in is not neutral. The megakernel beats D by ~1.9 ms at both lengths; that combines Gomes's attention kernel (vs NxDI's) and fusion, and Task 1 found no fusion (sync/DMA) saving.
- **Hypothesis:** D's fixed cost comes from the cascaded-attention kernel and/or the fused-QKV kernel path not being tuned for trn2 (NxDI's Qwen3-MoE recipe for this kernel targets trn3 with `--internal-max-instruction-limit=15000000`, `qwen.py:545`).
- **Surprising:** D's E2E is ~7 s/run longer than B's although TKG accounts for only ~1.3 s of it (760 × 1.75 ms). Hypothesis: extra host-side work per step in this path. Not investigated.

Open: profile D (one capture, ~25 min) to locate the +36 µs/layer; compare D's attention kernel against the megakernel's attention subkernel in isolation.

### Golden reference reproducibility — Measured
Re-ran `golden_fp32.py` (fixing the router top-9 capture, which saved position 0 only in the first run; output values were unaffected): v1 vs v2 max-abs 3.5e-3, per-position rel-L2 ≤ 3.1e-4, 0/2,560 argmax flips. CPU fp32 is not bit-reproducible across thread schedules (952% vs 452% CPU), but the drift is ~1% of the builds' error vs golden (0.04). Router data: `~/runs/golden_v2/` (`router_top`/`router_ids` `[48, T, 9]`); comparisons above use v1.

## 2026-11-11/12 — Task 7 (day 2): sweep extended to 4096

```bash
bash ~/runs/sweep4096.sh   # run_job.sh sweep_{libmoe,nki}_s4096 ... --seq-len 4096 --on-device-sampling --top-k 1 --do-sample "" --output-logits
```
Both compiled, loaded and benchmarked (exit 0). Compile 1,491 s (lib MoE) / 1,413 s (megakernel); wall 77 / 79 min; the megakernel's attention accepted 4096 at LNC=2 (sharded prior 2048, a multiple of 128).

| seq_len | 256 | 512 | 768 | 1024 | 2048 | **4096** |
|---|---|---|---|---|---|---|
| Lib MoE TKG p50 (ms) | 10.21 | 10.38 | 10.33 | 10.66 | 10.84 | **11.20** |
| Megakernel TKG p50 (ms) | 10.23 | 10.19 | 10.11 | 10.23 | 10.23 | **10.44** |
| Speedup p50 | 1.00× | 1.02× | 1.02× | 1.04× | 1.06× | **1.072×** |
| Lib MoE / megakernel TKG p99 (ms) | 10.53 / 10.56 | 10.75 / 10.81 | 11.09 / 10.59 | 11.08 / 10.71 | 11.36 / 10.48 | **11.91 / 11.05** (1.078×) |

768 = Task 5 means (n=5); 4096 single runs. CTE p50 at 4096: 382.5 / 401.9 ms (same +19.4 ms megakernel offset as at every other length).

What it shows (Measured): the gain keeps growing with context but more slowly (1.06× → 1.07×). The megakernel is no longer flat: +0.21 ms from 2048 to 4096 (lib MoE +0.36). Consistent with Task 1: its attention reads the full `seq_len` KV every step, which at 4096 is ≈2 MB/layer/rank and starts to show. Hypothesis: beyond 4096 the megakernel's fixed full-cache read would cost more than XLA's bucketed attention at short positions.

## 2026-11-12 — Task 6 (day 2): noise floor, XLA / library MoE / megakernel at 768

```bash
bash ~/runs/noise10.sh xla 1        # timing run: 14:15 wall (> 8 min) -> 5 runs per config
bash ~/runs/noise5_all.sh           # per config: preshard.py, 5 x run_job.sh ... --save-sharded-checkpoint --skip-compile, delete shards
```
Pre-sharding (`preshard.py`): XLA 505 s, lib MoE 3,127 s, megakernel 2,587 s (peak RSS 112–121 GiB). Disk holds one pre-sharded set at a time, so runs are consecutive per config, not interleaved as on day 1. Per-run wall: first run per config 10.5–14.3 min (cold page cache), later runs 4.7–6.3 min. Deletions logged in `~/runs/deletions.log`.

| TKG p50 (ms) | r1 | r2 | r3 | r4 | r5 | mean | std | min | max |
|---|---|---|---|---|---|---|---|---|---|
| XLA | 14.985 | 15.078 | 15.013 | 15.081 | 15.226 | **15.076** | 0.093 | 14.985 | 15.226 |
| Library MoE | 10.315 | 10.298 | 10.318 | 10.314 | 10.306 | **10.310** | 0.008 | 10.298 | 10.318 |
| Megakernel | 10.209 | 10.190 | 10.116 | 10.122 | 10.131 | **10.154** | 0.043 | 10.116 | 10.209 |

| | XLA | Library MoE | Megakernel |
|---|---|---|---|
| TKG p99 mean ± std | 15.719 ± 0.151 | 11.047 ± 0.023 | 10.655 ± 0.097 |
| E2E p50 mean ± std | 15,889 ± 238 ms | 11,901 ± 41 ms | 11,855 ± 175 ms |

Day 1 vs day 2 (n=5 each):
| | Day 1 | Day 2 | Pooled n=10 |
|---|---|---|---|
| Library MoE | 10.328 ± 0.036 | 10.310 ± 0.008 | 10.319 |
| Megakernel | 10.113 ± 0.020 | 10.154 ± 0.043 | 10.134 |
| Gap | 0.215 ms (2.1%) | 0.156 ms (1.5%) | 0.185 ms (1.8%) |
| XLA | 14.91 (n=1) | 15.076 ± 0.093 | – |

What it shows (Measured):
- The megakernel is faster than lib MoE on both days with non-overlapping ranges (day 2: slowest megakernel 10.209 < fastest lib MoE 10.298). The gap is 1.5–2.1% at 768.
- Between-day drift (megakernel +0.041 ms, XLA +0.17 ms vs day-1 single run) is larger than within-day std, so a ~0.05 ms difference between configs measured on different days is not resolvable; claims need same-session, interleaved runs.
- Lib MoE over XLA: 1.462× (day 2 means). Megakernel over XLA: 1.485×.
- Surprising: XLA's within-day std (0.093) is ~10× lib MoE's (0.008).

## 2026-11-11/12 — Day 3, T0: CPU triage of first divergences

```bash
python tools/triage_gross.py   # ~/runs/golden_v2 + day-1 captures -> ~/runs/triage/triage.json
```
Gross = KL(golden‖build) ≥ 0.2 and KL(golden‖XLA) < 0.01 at the first-divergence position (the census threshold); 16 gross (7 lib MoE, 9 megakernel) and 21 benign first divergences.

| | Gross (16) | Benign (21) | All 2,560 generated positions |
|---|---|---|---|
| Best-matching golden position for the build's output (p−1 / p / p+1) | 5 / 6 / 5 | 0 / 21 / 0 | – |
| Golden router min-over-layers 8th−9th gap, median | 0.00144 | 0.00128 | 0.00127 (p10 0.00020, p90 0.00424) |
| Same, median percentile vs all positions | 0.55 | 0.50 | – |
| TKG bucket / near a bucket edge (±2) | all 128 / 0 | all 128 / 1 | – |

Strongest shift matches (KL(build@p ‖ golden@q)):
| Build | Position | q = p−1 | q = p | q = p+1 |
|---|---|---|---|---|
| megakernel | p13@14 | 11.17 | 11.70 | **0.06** |
| lib MoE / megakernel | p06@4 | 11.83 / 11.76 | 11.78 / 11.73 | **0.42 / 0.55** |
| lib MoE | p02@111 | 20.02 | 16.62 | **0.42** |
| megakernel | p02@5 | 11.44 | 6.98 | 1.69 |
| lib MoE / megakernel | p01@15 | 11.22 / 11.11 | 19.57 / 19.53 | 18.71 / 18.74 |

What it shows:
- **Measured:** in 4 of 16 gross cases the NKI output at position p nearly equals golden's output at p+1, i.e. a prediction as if XLA's token p were already in context (" of" after "a new", where golden predicts " pair" then " of"). The other gross cases match no neighbour (KL 4–20 at all three).
- **Measured:** gross errors are not at small router margins (median 55th percentile, same as benign and random), so they are not explained by near-tie expert routing in the golden (H2.1 not supported for these).
- **Measured:** all at absolute positions 8–120, inside TKG bucket 128, none at a bucket edge.
- **Hypothesis:** the "lead" cases suggest the step reads state belonging to a different token/slot (e.g. a stale or misindexed buffer) rather than accumulated rounding; the teacher-forced census and the single-step variant (T1) test whether the error depends on earlier decode steps.

## 2026-11-12 — Day 3, T1: teacher-forced census of every position

```bash
# XLA: free-running capture of 4 long prompts + teacher-forced self-check (shard on load)
bash ~/runs/run_tf.sh tf_xla xla ~/qwen-30b-a3b/traced_xla_trn2_229_s768_greedy ~/runs/tf/xla \
  --free ~/runs/val_long/prompts_long.json --tf main=$HOME/runs/val long=$HOME/runs/tf/xla/free
python tools/golden_fp32.py --dtype fp32 --inputs "$HOME/runs/val_long/expected_p*.pt" --out ~/runs/golden_long
# NKI builds: teacher-forced 20 short + 4 long, repeat pass, single-step at gross positions
bash ~/runs/run_tf.sh tf_<b> <b> <build> ~/runs/tf/<b> --tf main=$HOME/runs/val long=$HOME/runs/val_long repeat=$HOME/runs/val \
  --single main=$HOME/runs/golden_v2=$HOME/runs/val long=$HOME/runs/golden_long=$HOME/runs/val_long
python tools/census.py --builds libmoe nki     # -> ~/runs/tf/census.json, ~/runs/tf/<b>/census_rows.json
```
Harness (`tools/capture_tf.py`): the production `generate()` path, with the adapter's forward wrapped so the on-device sampled token is replaced by XLA's reference token at every step. Long prompts: 4 fixed cuts of the model's README/LICENSE, 520/545/570/595 tokens + 64 new (`~/runs/val_long/prompts_long.json`, sha256 761e9a4f26ef4d25…). Golden for the 20 short prompts: `~/runs/golden_v2`; long: `~/runs/golden_long`. All loads shard weights on load (no pre-sharded weights exist now).

### Harness self-check — Measured
XLA teacher-forced vs XLA free-running: **bit-identical on 20/20 short prompts (vs day-1 references) and 4/4 long prompts** (outputs and tokens).

### Census, 2,816 positions per build (2,560 short + 256 long) — Measured
| | Library MoE | Megakernel | XLA (same positions) |
|---|---|---|---|
| **Gross** (KL(golden‖build) ≥ 0.2 and KL(golden‖XLA) < 0.01) | **0 / 2,816** (short 0/2,560, long 0/256) | **0 / 2,816** (0/2,560, 0/256) | – |
| Shared gross positions | 0 | 0 | – |
| KL histogram [0,1e-3) / [1e-3,1e-2) / [1e-2,0.05) / [0.05,0.2) / [0.2,1) / ≥1 | 2,050 / 668 / 83 / 13 / 1 / 0 | 1,999 / 702 / 105 / 7 / 2 / 0 | 2,009 / 688 / 109 / 8 / 1 / 0 |
| Max per-position KL (at) | 0.565 (p00@6) | 0.858 (p03@92) | – |
| KL mean | 1.94e-3 | 2.28e-3 | 1.88e-3 |
| ρ over all positions (with = without gross) | **0.909** | **0.935** | 1 |
| top-1 = golden | 98.5% | 98.9% | – |
| In-process determinism (repeat pass, 20 short prompts) | 20/20 bit-identical | 20/20 bit-identical | – |
| Cross-process vs day-1 free-running, up to first divergence | 12/20 prompts, 901/955 positions identical | 10/20 prompts, 685/729 positions identical | XLA: 20/20 (day-2 Task 3) |
| Single-step variant | not run: no gross positions to test | not run | – |

**Every day-1 gross error fails to reproduce (Measured).** For each build, the positions where today's teacher-forced output differs from the day-1 free-running output (same context) are exactly the day-1 gross first divergences — p01@15, p02@111, p03@27, p05@112, p06@4, p17@27 (lib MoE); p00@32, p01@15, p02@5, p05@2, p06@4, p08@5, p10@21, p13@14 (megakernel) — plus short drifts before p07@45 (both) and in p13/p11. At every day-1 gross position today's output agrees with golden: KL(golden‖today) ≤ 0.013 vs 1.2–24.8 on day 1.

What it shows:
- **Measured:** in a fresh process, neither NKI build makes a single gross error over 5,632 teacher-forced positions; both are as accurate as XLA against fp32 golden (ρ 0.91 / 0.94) and bit-deterministic within a process.
- **Measured:** the day-1 gross errors were specific to the day-1 validation processes, not a deterministic function of the input context. The NKI builds are not bit-reproducible across processes; XLA is.
- **Hypothesis (two candidates, tested next):** (a) state left by the day-1 `validate.py` call sequence (it runs `logit_validation` — dozens of generate calls of varying length per prompt — before the next prompt's first pass), or (b) the day-1 processes loaded **pre-sharded** weights written at compile time, whereas today's shard on load.
- Consequence for T2: the bisect targets (p01@15, p06@4, p07@45, census gross) do not fail in a fresh process, so a per-layer bisect against golden needs the failure to be reproduced first (replay, below).

## 2026-11-12 — Day 3, T1 follow-up: replaying the day-1 validation sequence

```bash
PRESHARD=0 bash ~/runs/run_replay.sh replay_libmoe_onload libmoe ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_greedy ~/runs/replay/libmoe_onload
# replay_validate.py = validate.py writing to OUT_DIR; weights shard on load (day 1 used pre-sharded weights)
```

| Library MoE, first free-running pass per prompt | Result |
|---|---|
| Replay vs day-1 `actual_libmoe_pNN.pt` | **20/20 prompts bit-identical, 128/128 positions each**, including all 7 gross first divergences |
| Replay first divergences | identical to day 1 (p00@31, p01@15, p02@111, p03@27, p04 none, p05@112, p06@4, …) |

What it shows:
- **Measured:** the gross errors are deterministic and reproduce exactly when the same sequence of `generate()` calls is run in a fresh process. Pre-sharded vs shard-on-load weights is not the cause (the replay sharded on load), so the planned pre-shard replay (R2) was dropped.
- **Measured (with T1):** the same positions are golden-correct when each prompt is preceded only by one teacher-forced pass of the previous prompt (census). So the output of a decode step depends on what earlier generate calls in the process did, not only on the current prompt's tokens. XLA shows no such dependence (bit-identical across both histories).
- **Superseded (history probe, below):** history has no effect; the replay matched day 1 because it repeated the same free-running passes.
- **Hypothesis (refuted later the same night):** the NKI MoE TKG kernel (common to both NKI builds) reads device memory that a previous execution wrote — e.g. a scratch/`shared_hbm` buffer or a cache region that is read before being written for some routing patterns — so stale contents leak into later steps. `history_probe.py` narrows which prior call is responsible.

## 2026-11-12 — Day 3: the day-2 "gross single-step errors" were an analysis artifact (CORRECTION)

```bash
bash ~/runs/run_py.sh history_libmoe tools/history_probe.py libmoe <build> ~/runs/replay/history_libmoe.json --pairs 0:1:15,5:6:4
bash ~/runs/run_py.sh free_<b> tools/free_tokens.py <b> <build> ~/runs/replay/free_<b>.json   # fresh free-running, emitted tokens saved
python tools/tie_audit.py                                                                   # -> ~/runs/triage/tie_audit.json
```

**History probe (Measured):** teacher-forced p01 and p06 are bit-identical to a fresh run after every history tried — teacher-forced / free-running / long-prefill passes of the previous prompt, the full day-1 `logit_validation` sequence on it, and a repeat (12/12 conditions, 0 gross). The "history-dependent" hypothesis from the replay section is wrong: the replay reproduced day 1 because it repeated the same *free-running* passes, not because of earlier state.

**Free-running token audit (Measured):** saving the tokens each build actually emitted shows that at an exact top-2 tie in its own scores (e.g. p02 step 110: 33.25 vs 33.25) the on-device sampler emits a different token than the host argmax of the returned scores. XLA's emitted tokens also differ from XLA's own argmax at 27 of its 36 exact ties. Day 1/2 located "first divergence" by comparing argmaxes of returned scores, not emitted tokens, so when the emitted tokens split at a tie the context at the next step already differed — and that next step was reported as the divergence, with "identical context" assumed.

| Day-1 first divergences | Library MoE | Megakernel |
|---|---|---|
| "Gross" (day-2 class) | 7 | 9 |
| … with the emitted token already different at an earlier step (contexts not identical) | **7 / 7** | **9 / 9** |
| … that earlier step was an exact top-2 tie in the build's scores / in XLA's scores | 5 / 2 | 5 / 4 |
| "Benign" with identical contexts at the divergence | 11 / 12 | 8 / 9 |

What it shows:
- **Measured:** every day-2 "gross single-step error" (KL 0.9–25 nats, ρ at divergence 8–63) was a comparison of outputs computed on *different* contexts, after a tie-broken token split one or more steps earlier. Under identical contexts (teacher-forced census, T1) neither NKI build has a single gross position in 5,632 (max KL 0.56 / 0.86), and both match XLA's accuracy against fp32 (ρ 0.91 / 0.94).
- **Measured:** the "lead/lag" shift matches in T0 follow from this: the NKI output was a correct prediction for its own, already different, context.
- **Retracted:** day-2 Task 2 statements that "at the first-divergence position the context is still identical to XLA's", the gross/benign split, the "shared input-triggered defect in both NKI paths" hypothesis, and day-2 SUMMARY finding 2 / INTERIM section 2's gross-error claims. The pre-divergence ρ results (0.96 / 1.05) stand.
- **Method fix (Measured necessity):** first-divergence analysis must compare *emitted tokens*, not argmaxes of returned scores; teacher forcing (T1 harness) removes the issue entirely. Exact bf16 top-2 ties are common (36 in XLA's 2,560 steps) and the on-device sampler's tie-break differs from `argmax`.

## 2026-11-12 — Day 3, T2: per-layer comparison against golden (was "bisect")

No position fails under identical context (T1 census, correction above), so there is no defect to bisect. T2 was run as the planned per-layer table at the named targets plus the largest-KL census positions: p01@15, p06@4, p07@45, p00@6, p17@41 (largest NKI-specific KL, shared by both builds), p03@92, p15@86, p23@4.

```bash
python tools/golden_fp32.py --dtype fp32 --inputs "$HOME/runs/val/expected_p*.pt" "$HOME/runs/val_long/expected_p*.pt" \
  --out ~/runs/bisect/golden --dump 1:15,6:4,7:45,0:6,17:41,23:4,3:92,15:86          # per layer: layer_in, attn_out, moe_in, router[128], moe_out
bash ~/runs/run_py.sh cap_libmoe tools/capture_layers.py libmoe ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_capture ~/runs/bisect/libmoe --targets …
bash ~/runs/run_py.sh cap_xla    tools/capture_layers.py xla    ~/qwen-30b-a3b/traced_xla_trn2_229_s768_capture    ~/runs/bisect/xla    --targets …
python tools/bisect_compare.py --builds libmoe=$HOME/runs/bisect/libmoe xla=$HOME/runs/bisect/xla   # -> ~/runs/bisect/bisect.json
```
Capture builds: NxDI `tensor_capture_config` with `modules_to_capture` = per layer `input_layernorm`, `self_attn.o_proj`, `mlp`, `capture_inputs=True` (288 tensors per decode step); compile 1,003 s (lib MoE) / 1,043 s (XLA). The fused NKI path never calls the router module, so router top-8 is recomputed on CPU (fp64, HF gate weights) from each build's captured MoE input; on the fused path `mlp.in` is the raw residual and is RMS-normalized on CPU, on the XLA path it is already normalized.

Checks (Measured): the lib MoE capture build's outputs are bit-identical to the non-capture build at all targets (capture does not change scheduling/numerics). Capture only returned data in TKG bucket 128: p23@4 (position 599, bucket 768) returned NxDI's `(1,)` fallback tensors for every module and is excluded.

| Target | lib MoE: max residual rel-L2 / layers with router top-8 ≠ golden | XLA: same | Max lib MoE / XLA ratio (attention out / MoE out) | First layer with ratio > 10× |
|---|---|---|---|---|
| p00@6 | 0.105 / 16 | 0.074 / 8 | 9.7 / 7.1 | none |
| p01@15 | 0.058 / 4 | 0.062 / 5 | 2.5 / 10.1 | L7, MoE output |
| p03@92 | 0.034 / 2 | 0.043 / 1 | 4.1 / 16.3 | L10, MoE output |
| p06@4 | 0.011 / 0 | 0.046 / 2 | 1.2 / 1.2 | none |
| p07@45 | 0.015 / 0 | 0.057 / 4 | 1.2 / 1.4 | none |
| p15@86 | 0.135 / 15 | 0.087 / 20 | 2.2 / 2.4 | none |
| p17@41 | 0.084 / 8 | 0.066 / 3 | 3.3 / 12.4 | L15, MoE output |

At the three jumps (rel-L2 vs golden):
| Target, layer | lib MoE attn / MoE in / MoE out | XLA attn / MoE in / MoE out | lib MoE router top-8 (recomputed) | Golden 8th−9th router gap at that layer |
|---|---|---|---|---|
| p01@15, L7 | 0.027 / 0.011 / **0.156** | 0.032 / 0.013 / 0.015 | 8/8 | **0.00008** (smallest of 336 layer-positions; median 0.065) |
| p03@92, L10 | 0.018 / 0.012 / **0.212** | 0.018 / 0.011 / 0.013 | **7/8** | 0.00403 (4th percentile) |
| p17@41, L15 | 0.047 / 0.044 / **0.213** | 0.027 / 0.017 / 0.021 | **7/8** | 0.03505 (33rd percentile; input already 2.6× XLA's error) |

What it shows:
- **Measured:** wherever lib MoE departs from golden by more than 10× XLA's error, it is the MoE-block output of a single layer, with that layer's attention output and MoE input no worse than XLA's (ratio ~1). The error then propagates (next layer's residual 4–5× XLA's) and decays/accumulates like XLA's own.
- **Measured:** at two of three jumps the router recomputed from lib MoE's own MoE input selects a different 8-expert set than golden (one swap); at the third the golden router margin is the smallest in the sample (8e-5). Positions with the largest final KL (p00@6, p15@86, p17@41) have router-set differences in 8–16 layers for lib MoE; the cleanest (p06@4, p07@45) have none.
- **Hypothesis (H2.1 supported):** these are expert swaps at near-tie router margins — the dominant mechanism of lib MoE's larger single-layer departures — not a kernel defect; the p01@15 L7 case suggests the kernel's own (lower-precision) routing flips where an fp64 recompute does not. Not tested: the kernel's internal router logits (not exposed by module capture).
- **Not run:** megakernel per-layer capture (its 48 layers are one NKI kernel; module-level capture has no per-layer modules to hook). The standalone-kernel fallback (step 4) was not needed.

## 2026-11-12 — Day 3, T3: profile of config D (lib MoE + NxDI NKI attention), seq 768

```bash
bash ~/runs/prof/run_prof.sh prof_cfgD_s768 cfgD ~/qwen-30b-a3b/traced_cfgD_trn2_229_s768_greedy 768   # profile_decode.py (cfgD kind), NEFFs dumped in-process
python tools/profile_table.py ~/runs/prof/tables/cfgD_s768.json cfgD_s768_b128=…bucket0.neff:… cfgD_s768_b768=…bucket3.neff:…
neuron-profile view … --output-format json   # least-skewed rank, bucket 128, DMA time by source file
```
Same method as day-2 Task 1 (inspect `model` mode, TKG calls 6 and 758, least-skewed rank; 970 MB raw, deleted after extraction).

| µs per layer (bucket 128 / 768) | Config D | Lib MoE (day 2) | Δ |
|---|---|---|---|
| Wall | 219.2 / 230.4 | 193.5 / 203.6 | **+25.7 / +26.8** |
| Sync engine | 59.3 / 59.1 | 60.2 / 60.1 | −0.9 / −1.0 |
| DMA active | 146.0 / 151.3 | 126.1 / 128.3 | **+19.9 / +23.0** |
| Tensor engine | 66.2 / 65.6 | 62.5 / 62.8 | +3.7 / +2.8 |
| Vector / Scalar / GpSimd | 52.8 / 36.2 / 60.6 (b128) | 45.5 / 55.3 / 61.3 | +7.3 / −19.1 / −0.7 |
| CC ops | 20.7 / 20.7 | 19.2 / 20.0 | +1.5 / +0.7 |
| DMA packets / avg bytes | 41,125 / 802 | 42,261 / 790 | −3% / +2% |
| Spill reload / save (bytes per graph) | 470,912 / 233,480 (b768: 556,928) | 212,864 / 122,888 | **2.2–2.6× / 1.9×** |
| Graph time (ms) | 10.52 / 11.06 | 9.29 / 9.77 | +1.23 / +1.29 |

DMA time by source (config D, bucket 128, µs/layer and MB/layer): MoE gate/up 40.0 / 12.6, MoE down 20.5 / 6.3, **NKI output projection (`output_projection_tkg.py`) 10.2 / 4.19**, untagged 7.2 / 3.2, **NKI attention block (`attention_block_tkg.py`) 5.8**, router 1.6 / 1.0, **NKI QKV (`qkv_tkg.py`) 1.3 / 5.24**.

What it shows:
- **Measured:** config D's extra device time is ~26 µs/layer (1.2–1.3 ms per step), mostly DMA-active time (+20–23 µs/layer); spills double; scalar-engine time drops 19 µs/layer (the kernel replaces XLA's softmax/RoPE work). The context slope is the same as lib MoE's in-graph (+11 µs/layer from bucket 128 to 768).
- **Measured:** the attention kernels' own DMA sources total ~17 µs/layer (output projection 10.2 µs for its 4.19 MB of weights ≈ 410 GB/s, attention block 5.8, QKV 1.3).
- **Measured:** device graph time explains ~1.2 ms of config D's 1.75 ms benchmark gap at 768; Hypothesis: the remaining ~0.5 ms/step is host-side work in this path (also visible as config D's +7 s E2E).
- **Hypothesis:** the NxDI attention-block TKG kernel's weight loads (o_proj in particular) are not overlapped with compute on trn2 and its working set spills; XLA attention schedules the same weight traffic under other work. A per-source comparison with lib MoE was not possible (its raw profile was deleted on day 2).
- Not reconciled: the benchmark found config D flat from 768 to 2048 while this in-graph profile grows +11 µs/layer from bucket 128 to 768 (n=1 step per bucket).

## 2026-11-12/13 — Day 3b, Task R: router decisions vs fp32 golden at every position

```bash
# capture builds: return_router_logits (+ return_expert_index for XLA); MoE.forward wrapped at runtime to register
# router outputs as manual capture tensors; mlp output module-captured; TensorRegistry.clear patched in-process
bash ~/runs/run_py.sh rcap_libmoe2 tools/capture_router.py libmoe ~/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_router2 ~/runs/router/libmoe2
bash ~/runs/run_py.sh rcap_xla     tools/capture_router.py xla    ~/qwen-30b-a3b/traced_xla_trn2_229_s768_router     ~/runs/router/xla
python tools/golden_fp32.py --dtype fp32 --inputs "$HOME/runs/val/expected_p*.pt" "$HOME/runs/val_long/expected_p*.pt" \
  --out ~/runs/router/golden --dump 0:all,…,23:all --dump-keys moe_out              # golden MoE outputs at every generated position
python tools/router_census.py --builds libmoe=$HOME/runs/router/libmoe2 xla=$HOME/runs/router/xla   # -> ~/runs/router/router_census.json
```

### Capture coverage — Measured
| Build | Attempt | Valid positions (prefill / bucket 128 / 256 / 768) | Note |
|---|---|---|---|
| lib MoE `…_router` | 1 (no patch) | 20/24 / 2,336/2,336 / 0/204 / 0/252 | same limit as day-3 T2 |
| XLA `…_router` | 1 (patched) | 24/24 / 2,336 / 204 / 252 | all buckets |
| lib MoE `…_router2` | 2 (patched) | 24/24 / 2,336 / 204 / 252 | all buckets |
- **Cause of the day-3 bucket limit (Measured, source):** `model_base._get_captured_tensors` ends with `registry.clear()`, and `TensorRegistry.clear()` replaces the config with `CapturedModelInfo([], 10, False)`; every bucket traced after the first then has no modules to capture and returns NxD's `(1,)` fallbacks. Patching `clear()` in-process to keep the config (tensors only reset) fixes all buckets; no library file was edited.
- Capture builds' outputs are bit-identical to the non-capture teacher-forced runs on 24/24 prompts (both builds): capture does not change numerics.
- The fused NKI MoE path returns the kernel's router logits but no expert indices (NxD returns an empty placeholder). Returned router logits are bf16-rounded in both builds: XLA's `topk(returned logits)` equals its actual `expert_index` at only 128,830/135,168 layer-positions (95.3%), while its `expert_index` is a valid tie-broken top-8 of the returned logits at **135,168/135,168**. Flips are therefore counted **tie-tolerantly** (golden's top-8 not a valid top-8 of the build's returned logits); stricter variants are shown for reference.

### Flip rates, 2,816 positions × 48 layers — Measured
| % of layer-positions with top-8 set ≠ golden | Library MoE | XLA |
|---|---|---|
| **Tie-tolerant** (like-for-like) | **8.40** | **9.60** |
| Strict: lib MoE topk(logits) / XLA actual expert_index | 12.70 | 13.82 |
| Positions with ≥ 1 flipped layer (tie-tolerant) | 79.5 | 83.8 |
| Per-layer flip rate: L0 / max (layer) / mean | 1.07 / 13.3 (L34) / 8.40 | 1.07 / 14.7 (L35) / 9.60 |

**NKI − XLA (tie-tolerant): −1.21 pp overall; per layer −2.70 to 0.00 pp (no layer above 0 pp) → passes the spec's 0.5 pp router gate at every layer.** Against XLA's actual ids: −5.42 pp.

| Golden 8th−9th router-logit gap | Library MoE | XLA |
|---|---|---|
| Median at flips / over all layer-positions | 0.0162 / 0.0600 | 0.0164 / 0.0600 |
| AUC of (−gap) as flip predictor (tie-tolerant; strict) | 0.780; 0.829 | 0.786; 0.829 |
| Flip rate by gap decile, smallest → largest (%) | 27.9, 18.0, 11.9, 8.5, 6.1, 3.9, 2.8, 2.2, 1.5, 1.2 | 31.3, 20.8, 14.4, 10.3, 6.7, 4.7, 2.9, 2.3, 1.4, 1.2 |

| Share of MoE-output squared error (vs golden) carried by flipped layer-positions | Library MoE | XLA |
|---|---|---|
| Overall | **86.6%** (flipped = 8.4% of layer-positions) | **87.7%** (9.6%) |
| Per layer: median / min (layer) | 0.85 / 0.21 (L33) | 0.82 / 0.28 (L33) |

**Worst-KL position:** p00@6 (KL(golden‖lib MoE) 0.565, KL(golden‖XLA) 0.231 — the only position where both exceed 0.2), absolute position 12, TKG bucket 128. Tie-tolerant flips: lib MoE in 11 layers (17–21, 23, 36, 37, 42, 44, 46), XLA in 5 (17, 36, 39, 42, 44); golden's smallest router gap there is 0.0014 (layer 1).

What it shows:
- **Measured:** the library MoE kernel's routing agrees with fp32 golden *more* often than XLA's does (8.4% vs 9.6% of layer-positions; fewer flips at every layer) — it passes the 0.5 pp router gate with margin.
- **Measured:** router flips are where nearly all MoE error lives (≈87% of squared MoE-output error in ≈9% of layer-positions, both builds), and golden's router margin predicts them (flip rate 28–31% in the smallest-gap decile vs 1.2% in the largest). H2.1 holds for both builds alike.
- **Measured:** ~4 pp of XLA's own flips are tie-breaks between exactly equal bf16 router logits; XLA's router decision is effectively made on bf16 logits.
- **Hypothesis:** the worst-KL position is a router-flip cascade at a low-margin position (lib MoE flips in 11 layers vs XLA's 5 there), not a systematic defect — consistent with lib MoE's lower flip rate overall.
- Caveat: lib MoE's flips are inferred from its returned (bf16) router logits, not from the kernel's internal indices, which NxD does not expose; the tie-tolerant count is a lower bound and the strict count an upper bound for both builds.
