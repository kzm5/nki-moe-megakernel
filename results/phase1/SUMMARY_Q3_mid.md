# Q3 checkpoint summary — T1, T2, T3 (trn2, Nov 19, 2026)

trn2.3xlarge, SDK 2.29, TP=4, LNC=2, bs=1, seq 768, greedy on-device sampling. Queue `docs/queues/Q3.md`; details in `FINDINGS.md` (Q3 entries at the top), runs in `ledger.jsonl`, state in `QUEUE_STATE.md`. Claims are Measured unless marked Hypothesis. Times ET.

## Headline
- **T2: the fp32 router is not a lever on this stack.** B, C and XLA already compute router logits and top-k in fp32 (`router_config.dtype = float32`, fp32 PSUM/SBUF logits, `--auto-cast=none`). A bf16 CPU emulation shows what that already buys over bf16 logits (KL −21.6%, strict flips −2.98 pp). Keeping the router input in fp32 as well adds nothing. T2.2 was skipped by its rule.
- **T3: on trn2, most of what fusion saves is per-call setup that an unfused kernel can also avoid.** E (the megakernel's attention as a standalone kernel) spends ~20 of its +38.6 µs/layer over C rebuilding the attention mask and permuted RoPE tables on every call. The builders' element-strided DMAs run on software DGE and stall the attention block and the QKV weight load. Building them once per step (Eh) recovers **32.0 µs/layer (83%)** in the profile and is bit-identical to E. Benchmark (n=5, 768): **Eh 10.707 ms vs E 12.255 and C 10.373 — 82% of E − C recovered**; Eh is within noise of B (10.579), still 0.335 ms above C (> 2 × pooled std).

## T1 skills (done)
Verified fixes from `SKILL_AUDIT.md`, one doc-only commit (`2af2cf5`): `hwdge`, no `nki.jit(platform_target=…)`, venv `~/venv-neuron-229`, `nki.simulate` bf16 caveat, trn3-only MX ops, profiler JSON in ns, ~1.2 GHz clock, per-rank CC-core ids, least-skewed rank and LNC=2 semantics.

## T2 fp32 router (done; T2.2 skipped)
CPU emulation, bf16 model, router varied, scored against the frozen golden with the validator's own KL and tie-tolerant flip rule (7,264 positions × 48 layers; 16 × 512 perplexity).

| Router | mean KL | top-1 | ppl | flips (validator rule) | flips (strict) |
|---|---|---|---|---|---|
| bf16 logits | 2.43e-3 | 98.49% | 8.342 | 9.53% | 13.87% |
| **fp32 logits + top-k (= B, C, XLA today)** | **1.90e-3** | **98.82%** | 8.329 | 10.89% | 10.89% |
| fp32 + fp32 router input | 2.05e-3 | 98.61% | 8.342 | 10.65% | 10.65% |

- Decision rule: the validator-rule flip rate does not fall (+1.36 pp; bf16 creates exact ties that the rule forgives), so T2.2 is skipped. The kernel change it describes would be a no-op anyway.
- Plan discrepancy: Phase 3 / Q3 T2 treat fp32 routing as untested; it is in every build. Router flips come from upstream bf16 hidden-state error. **Hypothesis:** the levers left are upstream, e.g. an fp32 residual stream, or bf16 rounding of the expert-GEMM inputs (H2.2: 30% of XLA's KL).

## T3 the fusion mechanism (done)
Per-instruction attribution by kernel phase (`tools/phase_attrib.py`; least-skewed rank, bucket 128; C re-profiled because its Phase 0 NTFFs were deleted).

| µs/layer | E | Eh (hoisted) | C | E − C | Eh − C |
|---|---|---|---|---|---|
| total | 231.3 | 199.3 | 192.7 | +38.6 | +6.6 |
| idle (engines stalled) | 45.0 | 29.0 | 28.0 | +17.1 | +1.0 |
| QKV (incl. weight DMA) | 19.9 | 13.2 | 13.4 | +6.5 | −0.2 |
| mask + RoPE build | 7.0 | 0.0 | 1.0 | +6.0 | −1.0 |
| X load / Y store | 5.2 | 5.6 | 0.0 | +5.2 | +5.6 |
| MoE (lib vs vendored) | 103.5 | 103.1 | 97.1 | +6.4 | +6.0 |

- **Cause 1, per-call mask/RoPE build, ~20 µs/layer:** about 6 µs of builder instructions plus ~13–15 µs of engines waiting at `attention_block_tkg.py:859` (`_to_sbuf(cos/sin)`) while 16 DMA queues run the builders' software-DGE transfers.
- **Cause 2, slow QKV weight load (+6.5), is a consequence of cause 1:** DMA-queue contention. With the hoist, E's QKV DMA matches C's.
- What remains is the call boundary itself (X load / Y store, ~5.6) and the MoE kernel choice (~6).

Timing, interleaved separate processes, n=5 at 768 (`bench_q3_t3.json`):

| Config | p50 TKG (ms) | − E | − C (2 × pooled std) | − B (2 × pooled std) |
|---|---|---|---|---|
| B lib MoE | 10.579 ± 0.129 | −1.676 | +0.206 (0.225) | – |
| C megakernel | 10.373 ± 0.094 | −1.882 | – | −0.206 (0.225) |
| E | 12.255 ± 0.100 | – | +1.882 | +1.676 |
| **Eh** | **10.707 ± 0.086** | **−1.547** | **+0.335 (0.180)** | **+0.128 (0.218)** |

- The hoist recovers 1.547 of 1.882 ms/step (32.2 µs/layer; profile 32.0). The queue's "within 2 × pooled std of C" is not met. The statement that holds: **on trn2, ~82% of what fusion saves for this stage is per-call setup cost (software-DGE mask/RoPE builds, plus the QKV-load stall they cause), which an unfused kernel can also avoid; the last ~7 µs/layer (0.34 ms/step) is the per-call activation round trip and the MoE kernel choice.**

## Next
T4: the agent study (A1, then C1, 6 h each) on the megakernel's collectives, after a harness sanity iteration on the unmodified kernel.
