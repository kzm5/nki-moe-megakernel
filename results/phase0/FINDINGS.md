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
| 2026-11-11 | **XLA + library MoE, greedy (fair)** | on-device greedy, logits out | 10.30 ms | 10.98 ms | 63.62 | pending (Task 3) | `~/runs/xlamoe_greedy_s768.log` |
| 2026-11-11 | **Megakernel, greedy (fair)** | on-device greedy, logits out | 10.12 ms | 10.57 ms | 64.64 | pending (Task 3) | `~/runs/nki_greedy_s768.log` |

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
