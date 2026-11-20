# Daily notes (≤15 lines each)

## 2026-11-17 (Tue, written 21:30 ET)
- Done: T1 XLA bit-identical to Phase 0 refs 24/24; corpus 64 prompts + 16×512 ppl (T3); validator built, golden computed, **frozen 14:06 ET** with hook (T2); XLA error budget, all buckets (T4).
- Done: T5 **Gate 1 met** — XLA passes vs itself (bit-identical across processes); m1 fails (worst_pos), m2 fails (all rows), m4 fails (router); control m3 passes.
- Done: frozen validator on B and C — **C (megakernel) passes**; **B (lib MoE) fails worst_pos on one position** (p60@69, KL 0.23 vs XLA 0.0098, same top-1, router flips at near-tie). Candidate fix (top-1 condition) in FINDINGS; spec not changed — your call.
- Done: T11(a) optimizer skill found both planted flaws (sim bit-identical; device check pending); 5 verified skill errors. T10 trn3 script drafted.
- Findings: capture build ≠ production XLA at the 512 prefill bucket (8/64 prompts) → logits rows use production XLA as baseline; m1's chunked-einsum fault took 2.4 h to compile.
- In progress: T6 config E first compile + validator run (started 21:15 ET).
- Blockers: GitHub issues disabled on kzm5/nki-moe-megakernel (enabling was denied to me) → status lines in `results/phase1/STATUS.md`.
- Decisions by rule: none (no m1/m2/m4 passed; m3 passed).

## 2026-11-18 (Wed, 08:00 ET)
- Done: T6 **config E built** (new files `attn_e_nki.py`, `qwen_attn_e.py`; first compile OK, ~1 h of the 10 h box). T7: E passes op-boundary (median ρ 0.93/0.93/0.94) and router (−1.27 pp); fails worst_pos on 1 same-top-1 position (p17@41), same pattern as B.
- Done: E's attention matches XLA's accuracy (L0 rel-L2 0.0026 vs 0.0028); E capture build bit-identical to its production build.
- Done: T8 seq 768, n=5 interleaved: **B 10.49 ± 0.08, C 10.29 ± 0.02, E 12.14 ± 0.01 ms** (p50 TKG). E is 1.65 ms *slower* than B (≈ config D's 12.08); fusion E→C = 1.85 ms ≫ 2× pooled std (0.03).
- In progress: T8 2048 (round 1: B 10.89, C 10.32, E 12.34), then 4096 (~6 h); then T9 profile of E, T11(b).
- Decision taken: E's timings are reported although E fails worst_pos as frozen (B fails the same way); labelled conditional. Decision rule at 768 points to "fusion share is real → Phase 4 sweep back in scope", pending 2048; caveat: E pays 48 NKI call boundaries/step and uses B's −O1 flags (C −O3) — T9 profile will split device vs host.
- Blockers: none. Waiting on you: worst_pos spec decision (FINDINGS "Gate 1", candidate 1); GitHub issues disabled.
- 10:40 ET addendum — T8 2048 (n=3): B 10.96 ± 0.07, C 10.37 ± 0.10, E 12.33 ± 0.07. ~~Decision rule applied: fusion share ≈1.9 ms; Phase 4 sweep back in scope.~~ **Corrected (Kevin): C − E is not fusion's share (E lands near config D, not B); fusion's net gain is C vs B = 0.20 / 0.59 ms; sweep NOT in scope.**
- 23:10 ET — T9 done (E's +1.8 ms is device time); T11 done (both skills give right numbers, several verified wrong instructions; optimized kernel bf16 bit-identical on device). **T8b: null NKI call per layer costs 0.17 ms/step (≈3.5 µs/call) = 10% of E's +1.69 ms → not the mechanism; C's flags change nothing (−0.04 ms).** H2.6 done: NKI builds start 30% more accurate at L0, converge by L47; all damp after L33–34.
- 17:35 ET — T8 done (4096: B 11.12, C 10.40, E 12.56 ms; C vs B 0.20/0.59/0.72 ms at 768/2048/4096). Rescore: under v1.1 **B, C, E pass**, Gate 1 still met (m1/m2/m4 caught, m3 passes). Found that `*.json` is gitignored — 17 Phase 1 JSONs had never been pushed; force-added (4d83e8d). Deleted the 6 approved builds (9.8 GB). Now: T9 profile E, then T8b.
- 14:10 ET — Kevin's decisions applied: spec v1.1 (worst_pos needs a top-1 change) written and re-frozen with v1.0 kept; rescoring all builds under both after T8 4096; T8b (null NKI kernel per layer; E with C's flags) queued after T9.

## 2026-11-20 (Fri, 05:25 ET) — Q3 day 1
- Done: T1 skill fixes (`2af2cf5`). T2: fp32 router logits + top-k are already in B, C and XLA; bf16 CPU emulation vs bf16 logits: KL −21.6%, strict flips −2.98 pp, tie-tolerant flips +1.36 pp → **T2.2 skipped by rule** (would be a no-op).
- Done: T3.1 attribution E vs C (C re-profiled): ~20 of E's +38.6 µs/layer is the per-call mask/RoPE build (software-DGE DMAs stall the attention block and the QKV weight load). T3.2 Eh (built once per step): passes v1.1, bit-identical to E; **10.707 vs E 12.255 / C 10.373 / B 10.579 ms (n=5) → 82% of E − C recovered**, still +0.335 ms vs C (> 2σ), ≈ B.
- Done: SUMMARY_Q3_mid.md (T1–T3) — two days before the checkpoint.
- Done: T4 harness sanity on the unmodified megakernel (PASS, 10.311 ms); **A1 (setup A): 3 iterations, all pass, best 10.101 ms (harness n=2), first improvement after 2 h 20 min, 0 interventions.** C1 (setup C) running 10:21–16:21 UTC.
- Kevin's decisions applied: disk 678 GB (JSONs kept); C pre-sharded (59 GB); swappiness 10 (Kevin); `MK_LOAD_PRESHARDED=1` hook bit-identical 64/64 + 16/16, load 149 s vs 806 s sharding on load (2,049 s before swappiness 10; Eh's cold load 2,537 s).
- Next: C1 → decide on A2/C2 → final interleaved bench of the best variant vs C and B (n=5).
- Blockers: none.
