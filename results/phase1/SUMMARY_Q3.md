# Q3 summary — kernel optimization and the trn2 fusion mechanism (Nov 19–23, 2026)

trn2.3xlarge, SDK 2.29, TP=4, LNC=2, bs=1, greedy on-device sampling. Queue `docs/queues/Q3.md`; checkpoint `SUMMARY_Q3_mid.md` (T1–T3 detail); lab notebook `FINDINGS.md` (Q3 entries at the top); runs `ledger.jsonl`. Claims are Measured unless marked Hypothesis. Frozen validator spec v1.1 throughout.

## Result line
**V = the T4 collectives variant (C1 iter 2: all-gather + local fp32 sum per all-reduce, LNC exchange overlapped) passes v1.1, is bit-identical across processes, and runs 10.050 ms at 768: −1.7% vs C and −4.28% vs B. Gate 4 (≥ 5% over B) is not met at the validated length; at 2048/4096 V is −7.6% / −8.6% vs B, on builds of the same source that the 768-only validator cannot score.**

## Headlines
- **T2:** the fp32 router is already in B, C and XLA; it is not a lever on this stack. T2.2 skipped by its rule.
- **T3:** ~82% of what fusion saves for the attention stage is per-call setup (software-DGE mask/RoPE builds and the QKV-load stall they cause), which an unfused kernel can also avoid (Eh). The last ~7 µs/layer is the call boundary and the MoE kernel choice.
- **T4:** all five agent runs found a validated, deterministic improvement of 0.2–0.3 ms/step (harness) from two families of change: moving the per-AR LNC exchange to `gpsimd_dma` with the residual add fused in, and replacing the all-reduce by all-gather + local sum. Setup C (profile feedback) was not better than setup A at the end of the budget; its best results came from the all-gather family, which twice produced non-deterministic builds.

## Pareto table (768)
Accuracy rows from the frozen validator v1.1 vs the fp32 golden (7,264 teacher-forced positions; 16 × 512 perplexity). ρ (median per-layer error ratio vs XLA) and router flip rate need a capture build; the megakernel family has none, so they are not measurable for C and its variants. Latency: same-session interleaved p50 where marked; otherwise harness n=2 (other sessions, not directly comparable).

| Config | TKG p50 (ms) | Source | mean KL vs golden | KL ratio vs XLA | ρ median attn / moe / resid | router flips | ppl | run-to-run | v1.1 |
|---|---|---|---|---|---|---|---|---|---|
| XLA (reference) | – | – | 2.258e-3 | 1.000 | 1 / 1 / 1 | 9.52% | 8.393 | 64/64 | PASS |
| B lib MoE | **10.500 ± 0.013** | final, n=5 | 1.967e-3 | 0.871 | 0.907 / 0.909 / 0.901 | 8.11% | 8.379 | 64/64 | PASS |
| C megakernel | **10.224 ± 0.007** | final, n=5 | 2.172e-3 | 0.961 | n/m | n/m | 8.385 | 64/64 | PASS |
| **V = C1 iter 2** `35309c3` | **10.050 ± 0.015** | final, n=5 | 2.168e-3 | 0.960 | n/m | n/m | 8.384 | 64/64 + cross-process 7,264/7,264 | PASS |
| E (unfused attention + lib MoE) | 12.255 ± 0.100 | T3 session, n=5 (C there 10.373) | 2.185e-3 | 0.967 | 0.927 / 0.929 / 0.939 | 8.24% | 8.397 | 64/64 | PASS |
| Eh (E, mask/RoPE hoisted) | 10.707 ± 0.086 | T3 session, n=5 | 2.185e-3 (bit-identical to E) | 0.967 | as E | as E | 8.397 | 64/64 | PASS |
| C2 iter 2 `b092e45` | 10.035 | harness n=2 | 2.177e-3 | 0.964 | n/m | n/m | 8.383 | 64/64 + cross-process | PASS |
| A2 iter 4 `9767038` | 10.060 | harness n=2 | 2.172e-3 (= C) | 0.961 | n/m | n/m | 8.385 | 64/64 | PASS |
| C2r iter 1 `e07beb4` | 10.082 | harness n=2 | 2.172e-3 (= C) | 0.961 | n/m | n/m | 8.385 | 64/64 | PASS |
| A1 iter 3 `00b7b0b` | 10.101 | harness n=2 | 2.172e-3 (= C) | 0.961 | n/m | n/m | 8.385 | 64/64 | PASS |
| *C1 iter 1* `65486dc` | 10.034 | harness n=2 | 2.131e-3 | 0.943 | n/m | n/m | 8.369 | **0/64** (race → invalid) | PASS* |
| *C2r iter 2* `b3cdb3d` | 9.859 | harness n=2 | 2.272e-3 | 1.005 | n/m | n/m | 8.386 | **0/64** | PASS* |
| *C1 iter 3* `ede5298` | 9.900 | harness n=2 | 2.260e-3 | 1.000 | n/m | n/m | 8.371 | 64/64 | **FAIL** (worst_pos, 2 gross) |

\* v1.1's run-to-run row is non-blocking ("documented"); non-deterministic builds are reported but are not eligible (Kevin, Nov 20). Unmodified megakernel in the harness: 10.311 ms.

- Accuracy is flat across the deterministic collectives variants: the LNC/`gpsimd_dma` family is bit-exact with C; the deterministic all-gather variants differ from C by ±0.3% KL. B is the most accurate config (KL ratio 0.871) and the slowest of B/C/V; there is no latency–accuracy trade-off inside the megakernel family.

## Final interleaved bench: V vs C and B (Nov 23, `bench_q3_final.json`)

| Seq | n | B | C | V | V − C (2σ pooled) | C gain over B | **V gain over B** | Gate 4 (≥ 5%) |
|---|---|---|---|---|---|---|---|---|
| 768 | 5 | 10.500 ± 0.013 | 10.224 ± 0.007 | 10.050 ± 0.015 | −0.174 (0.024) | 2.63% | **4.28%** | **not met** |
| 2048 | 3 | 10.893 ± 0.008 | 10.323 ± 0.003 | 10.064 ± 0.003 | −0.259 (0.007) | 5.23% | **7.61%** | met† |
| 4096 | 3 | 11.089 ± 0.032 | 10.401 ± 0.034 | 10.139 ± 0.019 | −0.263 (0.055) | 6.21% | **8.57%** | met† |

† V at 2048/4096 is the validated source compiled at another `--seq-len`; the frozen `validate.py run` is 768-only, so those builds are not validated (as for B and C's sweep builds in T8). Plan discrepancy: Gate 4 names no sequence length, and C alone already clears 5% at 2048/4096.

- The collectives change adds 1.7–2.5 points of gain over B at every length, far above noise. V is almost flat in context (+0.09 ms 768 → 4096; C +0.18, B +0.59).

## Fusion-mechanism attribution (T3, µs/layer, profile)

| | E | Eh | C | E − C | Eh − C |
|---|---|---|---|---|---|
| total | 231.3 | 199.3 | 192.7 | +38.6 | +6.6 |
| idle (engines stalled) | 45.0 | 29.0 | 28.0 | +17.1 | +1.0 |
| QKV (incl. weight DMA) | 19.9 | 13.2 | 13.4 | +6.5 | −0.2 |
| mask + RoPE build | 7.0 | 0.0 | 1.0 | +6.0 | −1.0 |
| X load / Y store | 5.2 | 5.6 | 0.0 | +5.2 | +5.6 |
| MoE (lib vs vendored) | 103.5 | 103.1 | 97.1 | +6.4 | +6.0 |

Hoisting the mask/RoPE build to once per step recovers 32 of 38.6 µs/layer (profile) and 1.55 of 1.88 ms/step (bench); Eh − C stays 0.335 ms (> 2σ). T4 shrinks C's remaining collective cost: V's profile is 187.1 vs C's 192.7 µs/layer (−5.6 µs/layer ≈ 0.27 ms/step; bench −0.17 to −0.26 ms), of which idle −5.4 (28.0 → 22.6) and collective-phase rows −2.4 (12.5 → 10.1).

## Agent study (T4): setup A (compile + validator verdict + p50) vs setup C (+ validator rows + per-phase profile)
Same start commit (`7fd70dc`; `689a988` = + the pre-sharded-load hook, a no-op for agent builds), same prompt apart from the SETUP block, 6 h budget each, fresh general-purpose subagent with the nki-kernel-optimizer skill. Latencies are harness n=2.

| Metric | A1 | C1 | A2 | C2 (partial) | C2r |
|---|---|---|---|---|---|
| Iterations / compiles / compile failures | 3 / 3 / 0 | 3 / 3 / 0 | 4 / 4 / 1 | 2 / 2 / 0 | 2 / 2 / 0 |
| v1.1 passes (of compiled) | 3 / 3 | 2 / 3 | 3 / 3 | 2 / 2 | 2 / 2 |
| Non-deterministic passing builds | 0 | 1 | 0 | 0 | 1 |
| Best eligible (ms) | 10.101 | **10.023** | 10.060 | 10.035 | 10.082 |
| Fastest build overall (ms) | 10.101 | 9.900 (fails v1.1) | 10.060 | 10.035 | 9.859 (non-det.) |
| First validated improvement after start | 2 h 20 min | 3 h 32 min wall / 2 h 37 active (first eligible) | 1 h 50 min (in noise); clear 5 h 17 min | 1 h 20 min | 2 h 09 min |
| Device time | 5 h 53 min | 4 h 59 min | 5 h 15 min | 3 h 34 min | 4 h 20 min |
| Wall / idle / active | 6 h 00 / 0 / 6 h 00 | 5 h 17 / 55 min / 4 h 22 | 5 h 17 / 0 / 5 h 17 | 2 h 20 / ~2 min / ~2 h 18 (stopped by usage limit) | 4 h 28 / 0 / 4 h 28 |
| Guard-hook blocks on frozen paths | 0 | 1 (false positive) | 0 | 0 | 0 |
| Other blocked tool calls | 2 | 3 | 1 | – | 0 |
| Orchestrator interventions | 0 | 2 (resume after forced hand-back; deadline +55 min) | 0 | 2 (resumes after forced hand-backs) | 0 |
| Cross-run information | none seen | none seen | none seen | read base/1 feedback + baseline profile | none (C2 records hidden) |
| Mechanism | LNC swap on `gpsimd_dma` + fused residual | AG + local sum; per-stage AG buffers | fused residual, then `gpsimd_dma` swap (iter 4) | AG + local sum; QKV-weight prefetch | `gpsimd_dma` swap + fused residual; then AG + local sum |

- **Measured:** every run improved on C within its budget; the spread of best eligible results (10.023–10.101) is within harness noise (~±0.05 ms). Setup C's best (10.023) and A's best (10.060) differ by less than that.
- **Measured:** setup C runs used the profile to choose the all-gather mechanism (C1, C2, C2r iter 2; A runs never tried it); 2 of its 3 implementations were non-deterministic (C1 iter 1, C2r iter 2) and one C build failed v1.1. Setup A runs only made bit-exact changes and never failed the validator.
- Hypothesis: profile feedback steers the agent toward bigger, riskier changes (AG reformulation) that the verdict-only setup does not attempt; with a validator that gates determinism this does not convert into a better validated result in 6 h. n = 2–3 runs per setup, so this is a description, not a significant difference.
- Study limitations: `~/runs/agents/` was readable by every run (C2 read the baseline profile); the runtime's forced hand-back hit Monitor-waiting runs (C1, C2) but not background-Bash runs; C2 was cut short by a usage-limit outage (tooling, not a setup effect) and rerun as C2r with C2's records hidden.

## What failed
- C1 iter 1 and C2r iter 2: all-gather + local sum builds non-deterministic run to run (0/64). C1 iter 1 classified as a race (fixed-order sum cannot vary; dedicated buffers fixed it). C2r's own hypothesis: sub-64 B per-rank slices sharing an SBUF line (untested).
- C1 iter 3: single-rounding residual update fails worst_pos (2 gross positions).
- A2 iter 1: compile failure (NKI does not allow nested function definitions).
- T3: Eh does not reach C within 2σ (+0.335 ms).
- Tooling: runtime forced hand-backs (C1, C2); C2 stopped by the usage limit (Nov 20 23:29 UTC); frozen validator is 768-only, so long-context variant builds cannot be scored.

## Branches pushed (`mine`; not merged — Kevin merges kernel changes)
| Branch | Head | Content |
|---|---|---|
| `q3/t3-fusion` | `a66879a` | Eh: `attn_e_hoist_nki.py`, `ATTN_E_HOIST=1` in `qwen_attn_e.py` |
| `q3/t4-base` | `7fd70dc` | agent-study start commit (harness sanity) |
| `q3/t4-a1` | `00b7b0b` | A1 iters 1–3 |
| `q3/t4-c1` | `ede5298` | C1 iters 1–3; **V = `35309c3` (iter 2)** |
| `q3/t4-a2` | `9767038` | A2 iters 1–4 |
| `q3/t4-c2` | `b092e45` | C2 partial, iters 1–2 |
| `q3/t4-c2r` | `b3cdb3d` | C2r iters 1–2 |

All agent changes are gated by `MK_AGENT_VARIANT=1` (default off; upstream path unchanged).

## Next (for Kevin)
- Decide Gate 4's sequence length: V meets 5% at 2048/4096 but not at 768. A long-context validator run (device.py `--seq-len` + `validate.py score`) would need a spec change.
- Candidate merge: V (`35309c3`). A composition of V with the bit-exact `gpsimd_dma` family (A1/A2/C2r iter 1) is untested (Hypothesis: the mechanisms overlap, since C2r iter 2 stacked them, 9.859 ms, but non-deterministically).
