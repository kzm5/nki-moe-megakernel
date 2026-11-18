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
