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
