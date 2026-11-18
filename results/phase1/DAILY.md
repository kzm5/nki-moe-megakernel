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
