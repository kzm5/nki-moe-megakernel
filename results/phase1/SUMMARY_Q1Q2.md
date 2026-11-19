# Q1 + Q2 summary — Phase 1 validator and config E (trn2, Nov 17–19, 2026)

trn2.3xlarge, SDK 2.29, TP=4, LNC=2, bs=1, on-device greedy sampling, logits returned. Queue: `docs/queues/Q1Q2.md`. Details: `FINDINGS.md`; runs: `ledger.jsonl`; state: `QUEUE_STATE.md`; daily notes: `DAILY.md`. All claims Measured unless marked Hypothesis. Times ET.

## Headline
- **Gate 1 met** (spec v1.0 and v1.1): XLA passes against itself (bit-identical across processes); the three faults m1, m2, m4 are caught; the benign control m3 passes.
- **Under spec v1.1 all three NKI configs pass the frozen validator** — B lib MoE, C megakernel, E (megakernel attention, unfused) — and are slightly more accurate than XLA against the fp32 golden.
- **Fusion's net gain on trn2 is C vs B: 0.20 / 0.59 / 0.72 ms per decode step at 768 / 2048 / 4096.** Running the megakernel's attention as a separate NKI kernel (E) is 1.4–1.7 ms *slower* than B; the bare NKI call boundary explains only 10% of that (≈3.5 µs per call), and C's compiler flags none. The fusion-depth sweep is not in scope.

## Validator (T2/T3) and the XLA error budget (T4)
- `tools/validator/`: `device.py` (teacher-forced capture; per-layer attention / MoE / residual + router), `golden.py` (fp32 CPU golden, stats computed on the fly), `validate.py` (spec rows → one JSON), `mutants.py`, `corpus.py`. Frozen with hashes (`FROZEN.md`, `frozen.json`; v1 kept as `*_v1`) and a PreToolUse guard hook; spec v1.0 (`spec.json`) and v1.1 (`spec_v1.1.json`, worst position must also change top-1) are both frozen.
- Corpus: 64 prompts (4–690 tokens, 64–128 generated; prose, code, math, dialogue, lists, 4 languages, wikitext excerpts), 7,264 teacher-forced positions; perplexity on 16 × 512 wikitext-2 tokens. XLA greedy reproduces the Phase 0 references bit for bit (24/24).
- XLA baseline: router flip rate 9.52% (tie-tolerant), mean KL 2.26e-3, top-1 98.65%, perplexity 8.393 (golden 8.342).
- XLA residual error vs golden grows from 0.007 (L0) to 0.068 (L34) and damps to 0.051 at L47; attention/MoE block errors peak at L37 / L33 (0.20 / 0.25).

## Gate 1 (T5)

| Build | v1.0 | v1.1 | Caught by | Key numbers |
|---|---|---|---|---|
| XLA, 2nd process | pass | pass | – | ρ 1.000, bit-identical 64/64 |
| m1 bf16 down-proj accumulation | fail | fail | worst position (1, top-1 changes) | median ρ 1.02, KL ratio 1.04 — barely separable from m3 on aggregates |
| m2 no top-k renormalization | fail | fail | all rows | median ρ 8–11, KL ratio 301, ppl +166% |
| m3 reordered expert sum (control) | pass | pass | – | median ρ 1.01–1.02, KL ratio 1.03 |
| m4 8th → 9th expert at L24 | fail | fail | router (+2.10 pp; L24 +69.7 pp) | KL ratio 1.21, just under 1.25 |
| B lib MoE | fail (1 position) | **pass** | – | median ρ 0.90–0.91, router −1.40 pp, KL ratio 0.87 |
| C megakernel | pass | **pass** | – | KL ratio 0.96 (op-boundary/router not measurable) |
| E | fail (1 position) | **pass** | – | median ρ 0.93–0.94, router −1.27 pp, KL ratio 0.97 |

Candidate rows (not adopted): tighter layer-max ρ (≤ 1.3) separates m1 from m3 on aggregates but would fail B (1.35); a per-layer router gate needs ≥ 5 pp to stay above sampling noise.

## B / C / E and the fusion split (T6–T8, T8b)

p50 TKG latency, ms, interleaved separate processes (n = 5 at 768, 3 at 2048/4096):

| Seq | B lib MoE | C megakernel | E | C vs B (fusion's net gain) | E vs B |
|---|---|---|---|---|---|
| 768 | 10.493 ± 0.082 | 10.291 ± 0.023 | 12.141 ± 0.006 | −0.20 | +1.65 |
| 2048 | 10.962 ± 0.070 | 10.372 ± 0.096 | 12.327 ± 0.068 | −0.59 | +1.37 |
| 4096 | 11.116 ± 0.042 | 10.397 ± 0.009 | 12.564 ± 0.008 | −0.72 | +1.45 |

T8b (768, n = 3, same session): B 10.521 ± 0.086, C 10.358 ± 0.088, E 12.215 ± 0.069, **N = B + 48 null NKI calls 10.690 ± 0.039**, **F = E with C's flags 12.179 ± 0.038**.
- The C − E arithmetic split is **not** fusion's share (Kevin's correction): E lands near config D (NxDI's NKI attention, 12.08 ms), not near B.
- Call-boundary cost: N − B = 0.17 ms/step ≈ 3.5 µs per call = 10% of E − B → not the trn2 fusion mechanism. Compiler flags: F − E = −0.04 ms (noise).
- E's attention is as accurate as XLA's (L0 attention rel-L2 0.0026 vs 0.0028); E's residual error tracks B's within 0–5%.

## Profile of E (T9)
Per layer at bucket 128 / 768: E 231.3 / 231.3 µs vs B 193.5 / 203.6 and C 192.5 / 192.8; graph time E 11.10 ms vs B 9.29 / 9.77. E's penalty is device time: +14–17 µs DMA, +13 µs vector, +6 µs collectives, 1.7× B's spill, −18 µs scalar per layer. E (like C) attends over the full 768-slot cache in every bucket, so it has no context slope. **Hypothesis:** what fusion saves on trn2 is per-call state of the standalone attention kernel (weights and constants reloaded into SBUF, mask/RoPE scratch rebuilt per call, its own all-reduce) — testable only with a kernel change (Q3).

## Early-finish item H2.6 (error vs depth)
NKI builds start with ~30% less residual error than XLA at L0 (lib MoE's layer-0 MoE output is more accurate) and converge to XLA's error by L47 (ratio 0.98–1.00); every build peaks at L33–34 and damps 25–36% to L47.

## trn3 readiness (T10)
`scripts/trn3_bringup.sh` (+ `.md`, `tools/bench_only.py`), not run. Verify first: driver/runtime support for trn3 devices on the current AMI (dkms 2.28 / runtime 2.32.31); `--logical-nc-config 2` with `--target trn3` (neuronx-cc 2.24's help says the flag is trn2-only; kernels assume LNC=2); lib MoE kernels and nkilib 2.29_release on trn3; NTFF/engine names for the profiler. One-day plan: check → compile A@768 and B/C/E at 768/2048/4096 → interleaved bench → one profile each → validator (XLA trn3 baseline; golden reused).

## Skill audit (T11)
- **nki-kernel-optimizer (trn2):** blind run found both planted flaws (redundant HBM round trip, sequential loop); output bit-identical on device in bf16. Wrong: `dge_mode.hw_dge` (enum is `hwdge`), `nki.jit(platform_target=…)` (deprecated), venv path; misses that `nki.simulate` does not round at bf16; trn3-only MX ops listed without caveat. Speed-up not resolved (dispatch-bound wall time).
- **neuron-profile:** numbers identical to `tools/profile_table.py`; wrong on units (JSON timestamps are ns, not ps), clock (~1.2 GHz, not 1 GHz), CC-core id (12–15, not 8), output size; silent on per-rank skew and LNC=2.

## What failed or changed
- Day-3b data was missing from the restored volume (snapshot predates it).
- Validator capture: four failed attempts before the right module set (prefill outputs are sequence-sharded; `self_attn`'s single captured tensor is not the attention output → capture `self_attn.o_proj`).
- The XLA capture build differs from the production build at the 512 prefill bucket (8/64 prompts), so logits rows use the production build as baseline.
- m1's chunked-einsum fault took 2.4 h to compile.
- My first reading of T8 ("fusion share ≈ 1.9 ms, sweep back in scope") was wrong; corrected by Kevin.
- `git pull --rebase` cannot rewrite the read-only validator files → rule changed to `--no-rebase` (approved).
- `*.json` is gitignored repo-wide; the Phase 1 JSONs were not pushed until Nov 18 (`git add -f` now).
- Not done: H2.2 (accumulation precision in the CPU golden).

## Recommended next steps
1. Q3 (with kernel-edit approval): split E's ~31 µs/layer of standalone attention cost — hoist mask/RoPE builds to once per step, pre-load weights — to name the mechanism fusion removes on trn2.
2. Q3 fp32 router logits: router flips carry ~87% of MoE error (Phase 0); measure ρ, flip rate and latency (one Pareto point) with the frozen validator v1.1.
3. trn3 day: run `scripts/trn3_bringup.sh` after checking LNC=2 on trn3.
4. Fix the two skills' verified errors before the agent study relies on them.
