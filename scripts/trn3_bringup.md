# trn3 bring-up (T10; written, not run)

Script: `scripts/trn3_bringup.sh {check|compile|bench|profile|validate|all}`. Benchmarks go through
`tools/bench_only.py` because `main.py --mode evaluate_single` on trn3 also compiles and loads the XLA
baseline for `logit_validation` (doubles compile time and host RAM).

## What must change from the trn2 AMI (verify in this order)
1. **Instance and capacity.** Spring used `trn3pd98.3xlarge` (one chip). Confirm the region offers it; the AMI is in
   sa-east-1 and must be copied to that region beforehand.
2. **Driver/runtime.** The trn2 AMI has aws-neuronx-dkms 2.28.0, runtime-lib 2.32.31, collectives 2.32.28,
   tools 2.30.10.0 (held). Check that this driver/runtime pair supports trn3 devices (`neuron-ls` shows the device
   and the runtime loads a trn3 NEFF). If not, install the trn3-capable driver/runtime and record the change in the
   ledger before running anything; note that the tools hold (2.30) exists only to read NTFF v7.
3. **Compiler/SDK.** neuronx-cc 2.24.5133 (SDK 2.29) lists `--target trn3` (Measured, `neuronx-cc compile --help`).
   Its help says `--logical-nc-config` "is only valid for Trn2 targets" — verify that LNC=2 is accepted/meaningful on
   trn3 (Spring ran TP=4, LNC=2 on trn3 with SDK 2.27). If LNC must change, the megakernel (`N_PRGS = 2`,
   `--seq-len` multiple of 256) and the E kernel assume LNC=2 and need re-checking.
4. **nkilib / kernels.** Fork branch `2.29_release` is the only combination that works on trn2 (legacy MoE needs a
   field removed in 2.30). The SDK MoE TKG kernels are LNC=2 kernels that Gomes's report treats as trn3-specific;
   confirm `--moe-fused-nki-kernel-enabled` compiles for trn3 with this nkilib. `main.py` must not force a target:
   `--platform-target trn3` sets `NEURON_PLATFORM_TARGET_OVERRIDE` (the Phase 0 fix removed the hardcoded trn3).
5. **Profiler.** `profile_decode.py` / `profile_table.py` assume trn2 engine names and NTFF v7; check that the
   trn3 NTFF opens with tools 2.30 and that engine names map (trn3 is NeuronCore-v4).
6. **Not needed:** the golden (fp32 CPU) and the corpus/refs are hardware-independent; reuse `~/runs/p1/{golden,refs}`.
   The XLA baseline must be re-captured on trn3 (config A) because ρ and the KL/perplexity ratios are relative to
   same-hardware XLA.

## One-day run plan (~14 h of device time, one job at a time)
| Hours | Step | Exit |
|---|---|---|
| 0–1 | `check`; pre-read weights (`~/runs/p1/preread.sh`, 16 parallel readers) | trn3 visible, compiler has trn3, imports OK |
| 1–6 | `compile`: A@768; B, C, E @768/2048/4096 (≈20–40 min each incl. shard-on-load) | NEFFs present; failures logged and skipped |
| 6–9 | `bench`: interleaved B→C→E, n=5 @768, n=3 @2048/4096 | p50/p99 per config |
| 9–10 | `profile`: one per-layer decode profile each at 768 | Table-5 format per config |
| 10–13 | `validate`: XLA trn3 baseline, then B, C, E through the frozen validator (logits, worst position, perplexity; op-boundary/router only if capture builds are added) | report per config |
| 13–14 | push, ledger, summary | trn2-vs-trn3 fusion comparison table |

Fallbacks: if E fails to compile on trn3, run B and C only; if LNC=2 is rejected, record it and stop (the
kernels' sharding is LNC=2-specific).
