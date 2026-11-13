# Phase 0 summary, day 3 — Qwen3-30B-A3B decode on trn2 (2026-11-12)

trn2.3xlarge, TP=4, LNC=2, bs=1, SDK 2.29, aws-neuronx-tools 2.30.10.0, branch `phase0/trn2-setup`. Queue: [`QUEUE_DAY3.md`](QUEUE_DAY3.md); state: [`QUEUE_STATE.md`](QUEUE_STATE.md); details: [`FINDINGS.md`](FINDINGS.md) ("Day 3" sections); runs: [`ledger.jsonl`](ledger.jsonl); resuming: [`RESUME.md`](RESUME.md).

## Key tables

Teacher-forced census vs fp32 golden, 24 prompts (20 short × 128 + 4 long of 520–595 tokens × 64), identical context at every position (harness self-check: XLA teacher-forced = free-running bit-for-bit, 24/24 prompts):

| | Library MoE | Megakernel | XLA |
|---|---|---|---|
| Positions | 2,816 | 2,816 | 2,816 |
| **Gross** (KL(golden‖build) ≥ 0.2 and KL(golden‖XLA) < 0.01) | **0** (short 0, long 0) | **0** (short 0, long 0) | – |
| Shared gross positions | 0 | 0 | – |
| KL histogram [0,1e-3) / [1e-3,1e-2) / [1e-2,0.05) / [0.05,0.2) / [0.2,1) / ≥1 | 2,050 / 668 / 83 / 13 / 1 / 0 | 1,999 / 702 / 105 / 7 / 2 / 0 | 2,009 / 688 / 109 / 8 / 1 / 0 |
| Max per-position KL | 0.565 (p00@6; XLA there 0.231) | 0.858 (p03@92; XLA 0.028) | – |
| ρ over all positions (= without gross) | **0.909** | **0.935** | 1 |
| top-1 = golden | 98.5% | 98.9% | – |

Determinism:
| | Library MoE | Megakernel | XLA |
|---|---|---|---|
| In-process (repeat pass, 20 prompts) | 20/20 bit-identical | 20/20 bit-identical | – |
| Same call sequence, new process (day-1 validation replay) | 20/20 prompts bit-identical to day 1 | – | – |
| Teacher-forced target after 6 different histories | 12/12 bit-identical | – | – |
| Across processes (day 2) | – | – | 20/20 bit-identical |
| Free-running vs XLA tokens | splits at exact bf16 top-2 ties (on-device sampler ≠ host argmax) | same | XLA's own emitted ≠ argmax at 27 of 36 ties |

Single-step variant: not run — no gross positions exist under teacher forcing.

Day-1 "gross" first divergences, audited with emitted tokens:
| | Library MoE | Megakernel |
|---|---|---|
| Day-1 "gross" (day-2 class) | 7 | 9 |
| Context already different (emitted token split at an earlier step) | **7/7** | **9/9** |
| … split at an exact tie in the build's / XLA's scores | 5 / 2 | 5 / 4 |

Per-layer comparison (lib MoE vs XLA vs golden, 7 targets in TKG bucket 128):
| Target | Lib MoE max residual rel-L2 / router-set ≠ golden (layers) | XLA same | Max ratio lib MoE/XLA (attn / MoE out) | First layer > 10× | Golden 8th−9th router gap there |
|---|---|---|---|---|---|
| p01@15 | 0.058 / 4 | 0.062 / 5 | 2.5 / 10.1 | L7 MoE out (0.156 vs 0.015) | 0.00008 (min of 336) |
| p03@92 | 0.034 / 2 | 0.043 / 1 | 4.1 / 16.3 | L10 MoE out (0.212 vs 0.013), router 7/8 | 0.0040 (4th pct) |
| p17@41 | 0.084 / 8 | 0.066 / 3 | 3.3 / 12.4 | L15 MoE out (0.213 vs 0.021), router 7/8 | 0.035 |
| p00@6 | 0.105 / 16 | 0.074 / 8 | 9.7 / 7.1 | none | – |
| p15@86 | 0.135 / 15 | 0.087 / 20 | 2.2 / 2.4 | none | – |
| p06@4 | 0.011 / 0 | 0.046 / 2 | 1.2 / 1.2 | none | – |
| p07@45 | 0.015 / 0 | 0.057 / 4 | 1.2 / 1.4 | none | – |

Config D profile (lib MoE + NxDI NKI attention, seq 768, µs/layer, bucket 128 / 768): wall 219.2 / 230.4 vs lib MoE 193.5 / 203.6 (+26), DMA active +20 / +23, spill reload 2.2–2.6×, scalar −19; NKI output-projection kernel DMA 10.2 µs/layer for 4.19 MB.

## Top 3 findings

1. **The day-2 "gross single-step errors" were an analysis artifact, not an NKI defect.** Every one of the 16 sat one or more steps after the builds' emitted tokens had split from XLA's at an exact bf16 top-2 tie (the on-device sampler breaks ties differently from host argmax; XLA's own emitted token ≠ its argmax at 27 ties). Argmax-based divergence detection then compared outputs from different contexts. Under teacher forcing neither NKI build has a single gross position in 2,816 (Measured). Day-2 claims retracted in FINDINGS, INTERIM and SUMMARY_2026-11-12.
2. **Both NKI builds are as accurate as XLA against fp32 golden and bit-deterministic:** ρ = 0.91 (lib MoE) / 0.94 (megakernel) over every position, max per-position KL 0.56 / 0.86, 20/20 bit-identical repeats in-process, and the same call sequence reproduces day-1 outputs bit-for-bit (Measured).
3. **Where lib MoE does depart most from golden, it is one layer's MoE output at a near-tie router (H2.1):** the first layer with > 10× XLA's error is always the MoE output (attention and MoE input ≈ XLA's), at a layer where the recomputed router swaps one expert or the golden router gap is the smallest observed (8e-5). Positions with the largest final KL have router-set differences in 8–16 layers; clean ones have none (Measured; "expert swap at near-tie" Hypothesis for the 8e-5 case where the fp64 recompute does not flip).

## What failed / was not done

- **Single-step variant (T1.5) and the classic bisect (T2 as written):** no position fails under identical context, so there was nothing to bisect; T2 ran as a per-layer error-vs-depth comparison instead.
- **Megakernel per-layer capture:** not possible with module capture (all 48 layers are one NKI kernel).
- **Capture outside bucket 128:** the long-prompt target (p23@4, bucket 768) returned NxDI's fallback tensors.
- **Wrong intermediate hypotheses (corrected in FINDINGS):** "history-dependent state" (from the replay) — refuted by the history probe (12/12 identical); pre-sharded weights — refuted by the shard-on-load replay.
- **Config D per-source comparison with lib MoE:** lib MoE's raw profile had been deleted on day 2.
- Minor: census.py float-equality bug, bisect_compare normalization mismatch for XLA's `mlp.in` and the `(1,)` fallback — fixed before reporting.

## Recommended next 3 experiments

1. **Router-precision study (H2.1, Phase 2):** capture the NKI MoE kernel's own router logits/expert indices (manual tensor registration inside the kernel wrapper or `return_expert_index` on a custom wrapper) at the 7 targets plus all positions with KL > 0.05, and measure flip rate vs golden router margin for lib MoE and XLA; test whether fp32 router logits in the kernel remove the L7-type departures.
2. **Fix the validation gate (Phase 1):** replace argmax-based divergence with teacher-forced per-position KL/ρ (this harness) and an explicit tie rule; rerun on a larger, hashed corpus with long prompts in TKG buckets > 128 (and fix capture there) to get a statistically meaningful max-KL tail for both NKI builds.
3. **Attention kernel on trn2:** with the context-scaling gain attributed to NKI attention and config D's NxDI kernel 1.2 ms/step slower in-graph (DMA + spills, o_proj weight load not overlapped — Hypothesis), profile the megakernel's attention subkernel against `attention_block_tkg` at the same shapes to see what a tuned unfused NKI attention would recover — this decides whether Phase 4's fusion sweep is worth running on trn2.

## Day 3b — router decisions vs golden (Task R, 2026-11-12 19:40 → 22:45 ET)

| % of layer-positions (2,816 positions × 48 layers) with top-8 ≠ fp32 golden | Library MoE | XLA |
|---|---|---|
| Tie-tolerant (like-for-like; returned router values are bf16) | **8.40** | **9.60** |
| Strict (lib MoE topk of returned values / XLA actual expert ids) | 12.70 | 13.82 |
| **NKI − XLA** | **−1.21 pp overall; −2.70 … 0.00 pp per layer → passes the 0.5 pp gate at all 48 layers** | |
| Share of MoE-output squared error on flipped layer-positions | 86.6% | 87.7% |
| Golden-margin AUC for predicting flips; flip rate smallest → largest gap decile | 0.78; 27.9% → 1.2% | 0.79; 31.3% → 1.2% |
| Worst-KL position p00@6 (lib MoE KL 0.565, XLA 0.231; bucket 128) | flips in 11 layers | flips in 5 layers |

Coverage: all decode buckets (128, 256, 768) and every prefill, after patching NxDI's `TensorRegistry.clear()` in-process (it reset the capture config after the first bucket's trace — the cause of day 3's bucket-128-only capture). Capture builds are bit-identical to non-capture builds on 24/24 prompts. XLA's expert ids are valid top-8s of its returned values at 135,168/135,168 layer-positions (method check); the fused kernel returns no ids, so lib MoE flips are inferred from its returned values.

Findings: (1) the library MoE kernel's routing matches fp32 golden more often than XLA's (Measured); (2) router flips at small golden margins carry ~87% of MoE error in both builds — H2.1 holds equally for both (Measured); (3) ~4 pp of XLA's flips are exact bf16 ties (Measured).
Failed: lib MoE capture attempt 1 covered only prefill + bucket 128 (fixed in attempt 2). Next: expose the kernel's internal indices (or fp32 router values) to replace the inferred-ids bounds; test fp32 router logits in the kernel at the high-flip layers (L21–36).
