# Phase 1 findings (lab notebook, Q1+Q2 onward)

Queue: [`docs/queues/Q1Q2.md`](../../docs/queues/Q1Q2.md). State: [`QUEUE_STATE.md`](QUEUE_STATE.md). Ledger: `~/runs/ledger.jsonl` → [`ledger.jsonl`](ledger.jsonl). Times ET unless marked UTC.

## 2026-11-17 11:05 — State on start (Measured)

- Commit `ffeed5c` (branch `phase0/trn2-setup`); work in worktree `.claude/worktrees/q1q2`, pushed with `git push mine HEAD:phase0/trn2-setup`.
- Disk: 160 GB free of 484 GB (`/`); RAM 121 GiB available of 124; swap 64 GB `/swapfile`, 0 used.
- `neuron-ls`: trn2.3xlarge `i-0f6abedd480e1a691`, 1 device, 4 logical cores, LNC=2, 96 GB HBM, idle. Runtime-lib 2.32.31, collectives 2.32.28, dkms 2.28.0, tools 2.30.10.0 (same as Phase 0).
- `gh auth status`: logged in as the repo owner (scopes repo, workflow).
- Builds present: XLA / lib MoE / megakernel `*_s768_greedy`, `traced_{xla,xlamoe}_*_s768_capture`, config D s768/s2048, sweep builds s256–4096, lib MoE router attempt-1 build `traced_xlamoe_*_s768_router`. None have pre-sharded `weights/`.
- **Missing from the restored volume:** all Day-3b output — `~/runs/router` is empty, and `traced_xlamoe_*_s768_router2` and `traced_xla_*_s768_router` are absent. The snapshot predates Day 3b (~19:40 ET Nov 12). `~/runs/ledger.jsonl` had 84 lines; the repo copy (88, a strict superset) was restored over it (logged in `~/runs/deletions.log`). The Day-3b results survive in git (FINDINGS, SUMMARY_DAY3); router reruns need the capture builds recompiled.
- `~/runs/run_{py,tf,job}.sh` and `ledger_event.py` now point to this worktree instead of `session-rules`.

## 2026-11-17 11:25–11:45 — T1 restore check (Measured)

```bash
~/runs/p1/preread.sh ~/qwen-30b-a3b/hf_model ~/runs/golden_v2 ~/runs/golden_long ~/runs/val ~/runs/val_long ~/qwen-30b-a3b/traced_xla_trn2_229_s768_greedy
~/runs/run_py.sh p1_xla_plain tools/validator/device.py xla ~/qwen-30b-a3b/traced_xla_trn2_229_s768_greedy ~/runs/p1/val/xla_plain \
  --corpus results/phase1/corpus/corpus.json --free all --refs ~/runs/p1/val/xla_plain/free --tf --ppl --repeat
python tools/compare_refs.py ~/runs/p1/val/xla_plain/free ~/runs/val ~/runs/val_long --out ~/runs/p1/t1_compare.json
```
- Pre-read: a serial `cat` ran at ~13 MB/s on the lazily restored volume; 16 parallel 1 GiB `dd` readers reached ~95 MB/s (68 GB in 12.5 min). After pre-reading, the XLA build loaded in **245 s** (Phase 0: 20–35 min shard-on-load).
- **XLA greedy is bit-identical to the day-1/day-3 references on 24/24 prompts** (20 short + 4 long; sequences and every logit). Same driver/runtime as Phase 0 (runtime-lib 2.32.31, dkms 2.28.0); XLA `model.pt` sha256 `b88382adf79e11dc…`.
- Harness self-check on the new 64-prompt corpus: teacher-forced = free-running bit for bit, 64/64 prompts; in-process repeat 64/64 identical. Job: 17 min wall, peak RSS 122 GiB.
- The 64 free-running outputs are the frozen references `~/runs/p1/refs/expected_pNN.pt` (p00–p23 identical to the Phase 0 files).

## 2026-11-17 — T3 corpus (Measured)

`tools/validator/corpus.py` → `results/phase1/corpus/corpus.json` (deterministic, seed 1117):
| Set | Prompts | Prompt tokens | Generated | Source |
|---|---|---|---|---|
| p00–p19 | 20 | 5–26 | 128 | day-1 prompts |
| p20–p23 | 4 | 520–595 | 64 | day-3 long prompts |
| p24–p37 | 14 | 4–24 | 128 | hand-written: code, math, dialogue, lists; Spanish, French, German, Chinese |
| p38–p47 | 10 | 65–121 | 128 | hand-written: code review, word problem, dialogue, recipe, JSON, SQL, prose, email, shell, table |
| p48–p63 | 16 | 259–690 | 64–96 | wikitext-2-raw-v1 test excerpts (CC BY-SA), half with an instruction |
| ppl w00–w15 | 16 × 512 tokens | – | teacher-forced from token 0 | wikitext-2-raw-v1 test, 16 distinct articles, raw format |

Totals: 64 prompts, 7,264 teacher-forced positions (+ 8,176 perplexity tokens); ≤ 768 tokens each (max 754). Wikitext test downloaded via the HF datasets-server rows API (4,358 rows = the full split), sha256 `bbf94c53…`.

## 2026-11-17 12:45–13:00 — T2 capture build for the validator: four failed attempts (Measured)

Capture build `traced_xla_trn2_229_s768_vcap`: modules `layers.i.self_attn`, `layers.i.mlp`, `layers.i` (48 each) + manual router logits (+ XLA expert ids), `TensorRegistry.clear` patched during tracing so every bucket keeps the capture config. Compile 1,165 s.
- Attempt 1 (load 2,525 s, compile + shard-on-load): crashed at p06's prefill. On the CTE step the MoE output capture is **sequence-sharded across TP** (16 rows for a 64-token bucket), so "row plen−1 of the padded prefill" is wrong — and would have silently read the wrong row for prompts shorter than 17 tokens. Fix: op-boundary and router rows are scored on decode steps only (p ≥ 1); the logits row still includes p = 0.
- Attempts 2–4 (loads 626 / 306 / 536 s): crashed on the first decode step; attempt 3 rested on a wrong guess (that the hook returns every element of a tuple output) and is superseded. Attempt 4 dumped the step shapes: 240 tensors = 144 module + 96 manual, **one tensor per module** (Day 3 was right), but for `layers.i.self_attn` that tensor is `[1, 1, 128]` (a RoPE/KV-side element of its tuple output), not the attention output; `mlp` and the decoder layer give `[1, 1, 2048]`. Fix: capture `layers.i.self_attn.o_proj` (single output, the TP-reduced attention output, as on Day 3) and, for config E, an identity child module on its TKG attention output. The capture build was deleted and recompiled (attempt 5).

## 2026-11-17 13:40–14:06 — T2 validator: baseline captured, golden, frozen (Measured)

```bash
~/runs/run_py.sh p1_xla_base tools/validator/device.py xla ~/qwen-30b-a3b/traced_xla_trn2_229_s768_vcap ~/runs/p1/val/xla_base \
  --corpus results/phase1/corpus/corpus.json --refs ~/runs/p1/refs --tf --ppl --repeat --capture      # attempt 5: 47 min
~/runs/run_py.sh p1_golden tools/validator/golden.py --corpus … --refs ~/runs/p1/refs --out ~/runs/p1/golden --write-golden \
  --stats xla_base=$HOME/runs/p1/val/xla_base                                                          # 9 min, 12 GB RSS
python tools/validator/validate.py score --cand ~/runs/p1/val/xla_base --base ~/runs/p1/val/xla_base --out …/report_self.json
python tools/freeze_validator.py
```
- Capture valid at **every decode step in every bucket** (128/256/512/768; 7,200 positions); prefill steps excluded by design. In-process repeat 64/64 bit-identical.
- **The capture build is not bit-identical to the production XLA build on 8/64 prompts** (p48–p56, prompt lengths 259–499, i.e. the 512 context-encoding bucket): they differ from the prefill step on, max per-position KL between the two builds 0.22 (p54). All shorter and all 768-bucket prompts are identical. Day 3b's 24/24 identity did not cover this bucket. Consequence (in `spec.json`): logits, worst-position and perplexity rows use the production XLA build (`xla_plain`) as baseline; op-boundary and router rows use the XLA capture build. Capture-build vs production KL ratio 0.999, perplexity identical (passages prefill one token).
- New golden vs Phase 0 `golden_v2` on p00–p19: max rel-L2 5.1e-5, identical argmax (batching only).
- Scorer self-test (capture build vs itself): ρ = 1.000 at every boundary, Δflip 0, gross 0 → pass.
- XLA baseline numbers (64-prompt corpus): tie-tolerant router flip rate **9.52%** of 345,600 layer-positions (Phase 0: 9.60%); mean KL(golden‖XLA) **2.26e-3**, top-1 = golden 98.65%, max KL 0.83 (p42@87), KL histogram [<1e-3, <1e-2, <0.05, <0.2, <1, ≥1] 5,182 / 1,749 / 293 / 37 / 3 / 0; perplexity XLA **8.393** vs golden 8.342 (+0.61%) over 8,176 wikitext tokens.
- **Frozen 19:06 UTC**: hashes in `results/phase1/FROZEN.md` / `frozen.json`, files `chmod a-w`. PreToolUse hook (`tools/validator/guard_hook.py`, user settings) verified live: an Edit of `spec.json` and a `touch` of it were both denied. It is deliberately conservative: any Bash command that names a frozen path together with `>`, `cp`, `tee`, … is denied, including heredoc appends and commit messages containing `<e-mail>`; use the Edit tool and `git commit -F`.

## 2026-11-17 — T4 XLA error budget vs fp32 golden, all decode buckets (Measured)

`python tools/error_budget.py --builds xla=$HOME/runs/p1/val/xla_base --corpus … --out results/phase1/error_budget/xla` → `xla.json` / `xla.csv` (plot data: build, boundary, bucket, layer, rel-L2, cosine, max-abs, n). Pooled over 7,200 teacher-forced decode positions (bucket 128: 4,371; 256: 1,217; 512: 729; 768: 883). Statistics are computed on the fly against the golden forward; golden per-layer tensors are kept only at 512 seeded positions (`~/runs/p1/golden/layers.pt`).

| Layer | attn | MoE | resid | resid b128 | b256 | b512 | b768 | cos(resid) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0028 | 0.0165 | 0.0074 | 0.0071 | 0.0073 | 0.0076 | 0.0084 | 0.99997 |
| 4 | 0.0268 | 0.0470 | 0.0162 | 0.0164 | 0.0150 | 0.0176 | 0.0159 | 0.99987 |
| 8 | 0.0274 | 0.0581 | 0.0203 | 0.0203 | 0.0203 | 0.0214 | 0.0192 | 0.99979 |
| 12 | 0.0515 | 0.0854 | 0.0264 | 0.0269 | 0.0274 | 0.0254 | 0.0227 | 0.99965 |
| 16 | 0.0610 | 0.0821 | 0.0303 | 0.0317 | 0.0315 | 0.0262 | 0.0246 | 0.99954 |
| 20 | 0.0691 | 0.1073 | 0.0344 | 0.0354 | 0.0374 | 0.0290 | 0.0287 | 0.99941 |
| 24 | 0.1016 | 0.1439 | 0.0441 | 0.0447 | 0.0496 | 0.0387 | 0.0371 | 0.99903 |
| 28 | 0.0946 | 0.1261 | 0.0464 | 0.0470 | 0.0523 | 0.0390 | 0.0404 | 0.99892 |
| 32 | 0.1078 | 0.1731 | 0.0526 | 0.0537 | 0.0604 | 0.0422 | 0.0427 | 0.99862 |
| 36 | 0.1535 | 0.1611 | 0.0654 | 0.0677 | 0.0730 | 0.0511 | 0.0514 | 0.99786 |
| 40 | 0.1001 | 0.0966 | 0.0521 | 0.0506 | 0.0612 | 0.0488 | 0.0488 | 0.99864 |
| 44 | 0.0564 | 0.0832 | 0.0482 | 0.0455 | 0.0559 | 0.0457 | 0.0522 | 0.99884 |
| 47 | 0.0288 | 0.0497 | 0.0510 | 0.0492 | 0.0646 | 0.0436 | 0.0491 | 0.99870 |

Medians over layers: attention 0.068, MoE 0.096, residual 0.043; maxima attention 0.196 (L37), MoE 0.248 (L33), residual 0.068 (L34).
- **Measured:** residual-stream error grows roughly linearly with depth to L34–36 (0.007 → 0.065) and then declines to 0.051 at L47; attention and MoE block errors are 2–4× the residual error (their outputs are small relative to the residual).
- **Measured:** bucket dependence is small; the 256 bucket is 10–20% higher than the others in layers 20–47.
- **Hypothesis (H2.6):** the late-layer decline means the residual stream damps per-layer error after ~L36 rather than amplifying it; to be tested with the lib MoE and E captures.

## 2026-11-17 14:07 → — T5 device chain (`~/runs/p1/chain_q1.sh xla_self m1 m2 m4 m3 libmoe nki`) (Measured)
- `xla_self` (same XLA capture build, second process, 14 min): teacher-forced scores **bit-identical to the baseline process on 64/64 prompts**; in-process repeat 64/64.
- `m1` (bf16-accumulated expert down-projection, 24 chunks of 8 per rank): **compile 8,774 s (2.4 h)** vs 1,073 s for the unmutated capture build — the chunked einsum multiplies the TKG/CTE graph size; load 9,483 s total; repeat 64/64. Budget note: torch-level faults that expand the graph cost hours of compile.
- `m2` 34 min (compile 1,082 s), `m4` 35 min (1,191 s), `m3` 31 min (917 s), `libmoe` capture build 45 min (1,069 s), `nki` (Phase 0 greedy build, no capture) 11 min. All EXIT 0, every in-process repeat 64/64 bit-identical. m4 check: at layer 24 the captured expert set is not a valid top-8 of the router logits at 117/127 steps of p00 (the other 10 are exact 8th/9th ties); layers 23/25 unaffected.
- One golden `--stats` pass for xla_self, m1–m4, libmoe: 689 s, 31 GB RSS.

## 2026-11-17 21:20 ET — T5 Gate 1: frozen validator on XLA, mutants, lib MoE, megakernel (Measured)

```bash
bash ~/runs/p1/score_q1.sh m1 m2 m3 m4 libmoe nki      # validate.py score per build -> ~/runs/p1/reports/<build>.json
python ~/runs/p1/summarize_reports.py xla_self m1 m2 m3 m4 libmoe nki
```
All reports `frozen_ok: true`. Baselines: op-boundary/router = XLA capture build (`xla_base`); logits/worst/perplexity = production XLA (`xla_plain`).

| Build | Verdict | Failed rows | median ρ attn / MoE / resid (max) | Router Δ vs XLA (pp) | KL ratio | Gross | max KL | ppl Δ | Run-to-run |
|---|---|---|---|---|---|---|---|---|---|
| XLA, 2nd process | **pass** | – | 1.000 / 1.000 / 1.000 | 0.00 | 0.999 | 0 | 0.833 | 0.00% | bit-identical in-process and across processes (64/64) |
| m1 bf16 down-proj accumulation | **fail** | worst_pos | 1.018 / 1.016 / 1.021 (1.45 L33) | +0.10 | 1.043 | 1 (p47@3) | 0.961 | −0.28% | bit-identical |
| m2 no top-k renormalization | **fail** | all five | 9.36 / 8.01 / 11.37 (48.8) | +80.1 | 301 | 2,761 | 14.1 | +166% | bit-identical |
| m3 reordered expert sum (control) | **pass** | – | 1.016 / 1.014 / 1.013 (1.20) | +0.07 | 1.033 | 0 | 0.745 | −0.07% | bit-identical |
| m4 8th → 9th expert at L24 | **fail** | router | 1.000 / 1.000 / 1.000 (MoE 1.58 at L24) | **+2.10** (L24 +69.7) | 1.215 | 0 | 0.508 | −0.18% | bit-identical |
| B lib MoE (capture build) | **fail** | worst_pos | **0.907 / 0.909 / 0.901** | **−1.40** | **0.871** | **1 (p60@69)** | 0.564 | −0.17% | bit-identical |
| C megakernel (production build) | **pass** | – (op-boundary, router not measurable) | – | – | **0.961** | 0 | 0.858 | −0.10% | bit-identical |

**Gate 1 = met as specified:** XLA passes against itself; m1, m2, m4 all fail, caught by worst_pos (m1), every row (m2), router (m4); the control m3 passes.

What the table shows:
- **Measured:** m1 is caught only by one position. On every aggregate row it is nearly indistinguishable from the benign control m3 (median ρ 1.02 vs 1.01, KL ratio 1.043 vs 1.033, router +0.10 vs +0.07 pp); only the layer-max ρ separates them (1.45 vs 1.20, both under the gate of 2). Bf16 accumulation over 8-wide chunks adds little error relative to XLA's own bf16 error budget. The catch is fragile: at p47@3 m1 changes golden's top-1 (" |" 0.73 → " " 0.58, router flips in 11 layers), but m3 reaches KL 0.074 at the same position.
- **Measured:** m4 is caught by the router row (+2.10 pp overall, +69.7 pp at L24), and its KL ratio 1.215 is just under the 1.25 gate; op-boundary median ρ is 1.000 because only one layer is faulty (max MoE ρ 1.58 at L24 < 2).
- **Measured — the frozen validator fails library MoE (B) on the worst-position row** with one gross position, p60@69 (B KL 0.234; XLA 0.0098, just under the 0.01 clean threshold). There B keeps golden's top-1 (" contributing", golden 0.65, B 0.87, XLA 0.69), flips the router at L37 and L39 (XLA at L37; golden L37 margin < 0.005); the megakernel sharpens the same way (KL 0.152). Every other row passes with margin (ρ ≈ 0.90, −1.40 pp flips, KL ratio 0.87). Phase 0's census (2,816 positions, 24 prompts) did not contain this prompt. **Hypothesis:** a near-tie router-flip cascade, not a kernel defect (same mechanism as Phase 0's p00@6).
- **Measured:** the megakernel passes every measurable row on the new corpus.

Candidate rows (not added to the frozen spec):
1. **Worst position with top-1 condition:** gross = cand KL > 0.2 and base KL < 0.01 **and cand top-1 ≠ golden top-1**. On these data it keeps m1 caught (p47@3 changes top-1) and clears B's p60@69 (same top-1). It also removes the knife-edge at the base KL threshold (XLA 0.0098 vs 0.01).
2. **Layer-max ρ at a tighter bound** (e.g. ≤ 1.3 per boundary): separates m1 (1.45) from m3 (1.20) on aggregate statistics instead of one position; check false positives on lib MoE (1.35 at L33) before adopting — it would fail B.
3. **Per-layer router Δ**: catches m4 sharply (+69.7 pp at L24; lib MoE's per-layer max Δ is 0.00 pp), but at 0.5 pp per layer it would also fail the control m3 (+0.79 pp at L19) and m1 (+0.94 pp at L32) — sampling noise at ~7,200 positions per layer is ~0.3–0.5 pp; a per-layer gate needs a larger threshold (e.g. 5 pp).

Flag for Kevin: as frozen, the spec fails the Phase 0 reference config B on one position; the decision (keep, or adopt candidate 1 in a new spec version) is his.

**Spec v1.1 adopted (Kevin, 2026-11-18; re-frozen 19:06 UTC / 14:06 ET).** `results/phase1/spec_v1.1.json` = v1.0 plus `gross_requires_top1_change: true`: a worst-position counts only if cand KL > 0.2, base KL < 0.01 **and** cand top-1 ≠ golden top-1. Reason: under v1.0, lib MoE (p60@69) and E (p17@41) each failed on a single near-tie position where they keep golden's top-1 but are more confident, and the "clean" threshold sat on a knife edge (XLA 0.0098 vs 0.01); m1's gross position changes top-1 and stays caught. Timing: decided after the T5/T7 results; all v1.0 numbers above stand as reported under v1.0. Implementation: `validate.py --spec {1.0,1.1}` (default 1.1); `frozen_ok` requires the spec's hash to be one of the frozen versions. Check (Measured): rescoring lib MoE under v1.0 with the new code reproduces the original v1.0 report row for row. v1 freeze kept as `FROZEN_v1.md` / `frozen_v1.json`. Hardening: the freeze records themselves (`frozen*.json`, `FROZEN*.md`) and both specs are now guarded paths; under v1 the guard read an unprotected `frozen.json`, which an agent could have moved to disable it (verified closed: moving `frozen.json` and editing `spec_v1.1.json` are now denied).

**Rescore of every build under v1.0 and v1.1 (17:06–17:32 ET, Measured; `tools/chain_q2b.sh rescore` → `results/phase1/reports_v10/`, `reports_v11/`; all `frozen_ok: true`).** m1–m3's per-layer captures were deleted for disk before the rescore, so their router row reads "not measurable"; it is identical under both versions and carries over from the original v1.0 reports (m1 +0.10 pp pass, m2 +80.1 pp fail, m3 +0.07 pp pass).

| Build | v1.0 verdict (failed rows) | v1.1 verdict (failed rows) | gross v1.0 → v1.1 |
|---|---|---|---|
| XLA 2nd process | pass | pass | 0 → 0 |
| m1 bf16 down-proj accumulation | fail (worst_pos) | **fail (worst_pos)** | 1 → 1 (p47@3 changes top-1) |
| m2 no renormalization | fail (all; router carried over) | **fail** (all) | 2,761 → 1,373 |
| m3 reordered sum (control) | pass | **pass** | 0 → 0 |
| m4 8th→9th expert at L24 | fail (router) | **fail (router)** | 0 → 0 |
| B lib MoE | fail (worst_pos) | **pass** | 1 → 0 |
| C megakernel | pass | **pass** (op-boundary/router not measurable) | 0 → 0 |
| E (production build) | fail (worst_pos) | **pass** (op-boundary/router not measurable) | 1 → 0 |
| E (capture build) | fail (worst_pos) | **pass** (all rows measured) | 1 → 0 |

- **Gate 1 stays met under v1.1** (XLA passes vs itself; m1, m2, m4 caught by the same rows; m3 passes).
- B, C and E all pass the frozen v1.1 validator; **T7 is satisfied, so E's T8 timings are no longer conditional.**

Incident (14:20 ET, Measured): the required `git pull --rebase mine phase0/trn2-setup` (remote had one new commit, RESEARCH_PLAN rev 35) failed with "unable to unlink tools/validator/validate.py: Permission denied" — a rebase must rewrite files in the read-only validator directory — and left the working tree half-switched to the remote tip (my commit `96e49f2` intact). Recovery without touching the freeze: verified the working-tree changes were byte-identical to the remote tip, stashed the non-frozen ones (`q1q2-stale-remote-tip-snapshot-20261118`, `5848957e`), merged instead of rebasing (`git pull --no-rebase`, merge `3cb5c40`; only `docs/RESEARCH_PLAN.md` came in; push fast-forward). The guard correctly refused to let me rewrite the two frozen records on disk; local `results/phase1/FROZEN.md` / `FROZEN_v1.md` are stale copies (the committed versions are correct) and are never staged. **Rule conflict for Kevin:** with a read-only validator, `git pull --rebase` fails whenever local commits touch frozen files; merge pulls (or rebasing only when no local commit touches them) are needed.

## 2026-11-17 21:15–22:00 ET — T6 config E built; T7 validator on E (production build) (Measured)

E = XLA + library MoE kernel + the megakernel's TKG attention stage per layer, unfused. Code (new files only, no kernel edits): `megakernels/qwen3_moe/attn_e_nki.py` (one `nki.jit` kernel per layer: shard-interleaved X load, vendored `nki_kernels._legacy.attention.attention_block_tkg` with fused input RMSNorm, pre-RoPE QK RMSNorm, RoPE, in-place K/V update at `position_ids`, output projection, `_sb2sb_all_reduce_gather`; helpers imported from `transformer_qwen3_moe_speculative.py`, the module build C uses) and `megakernels/qwen3_moe/qwen_attn_e.py` (subclass of `qwen.py`: TKG self-attention dispatches the kernel, residual adds and the MoE block stay in XLA; weight conversion adds the megakernel's rank-grouped fused QKV and Wo^T; neuron-config flags as the megakernel: `attn_tkg_nki_kernel_enabled`, `attn_block_tkg_nki_kernel_cache_update`; compiler flags = `qwen.py`'s (B's) + `--enable-verifier=false` for the in-place KV scatter). Selected with `--qwen megakernels.qwen3_moe.qwen_attn_e --moe-fused-nki-kernel-enabled` (validator kind `E`). `main.py` needed no change (`--qwen` already selects a module).

- **Compiled on the first attempt** (884 s; all HLOs PASS) and loaded (1,099 s incl. shard-on-load). Time-box used: ~1 h of 10.
- Note (Hypothesis, confound for T8): E uses B's compiler flags (TKG `-O1`, `cc-pipeline-tiling-factor=2`); C uses its own (`-O3`, tiling 4, extra DMA vectorization). C − E therefore mixes fusion with compiler-flag effects.

Frozen validator on E (production build, logits rows; op-boundary/router pending the E capture build):

| | E | C megakernel | B lib MoE | XLA |
|---|---|---|---|---|
| Verdict | **fail** (worst_pos) | pass | fail (worst_pos) | – |
| mean KL vs golden (ratio to XLA) | 2.185e-3 (**0.967**) | 2.172e-3 (0.961) | 1.967e-3 (0.871) | 2.260e-3 |
| top-1 = golden | 98.65% | 98.73% | 98.68% | 98.65% |
| gross positions | **1 (p17@41)** | 0 | 1 (p60@69) | – |
| perplexity (Δ vs XLA) | 8.397 (+0.05%) | 8.385 (−0.10%) | 8.379 (−0.17%) | 8.393 |
| run-to-run | bit-identical (64/64) | bit-identical | bit-identical | bit-identical |

E's gross position p17@41 is a two-way near tie in golden ("," 0.60 / " line" 0.40). Every build keeps golden's top-1: XLA KL 0.0011 (0.62); lib MoE, megakernel and the benign control m3 all 0.129 (0.82); E 0.328 (0.905). It is the position Phase 0 called "the largest NKI-specific KL, shared by both builds". Same pattern as B's p60@69; candidate worst-pos row 1 (top-1 must change) would clear both.

**Decision taken (rule interpretation, for Kevin):** T7 says E must pass before any timing claim. As frozen, E fails — and so does the reference config B — on a single same-top-1 position. T8 measurements proceed; every E timing claim is labelled "conditional on the validator verdict (fails worst_pos on 1 same-top-1 position)".

E capture build (`traced_E_trn2_229_s768_vcap`, compile 893 s, captures via an identity module on the kernel's attention output): **bit-identical to the E production build on 64/64 prompts**; valid at every decode step. Full validator: op_boundary **pass** (median ρ attention 0.927, MoE 0.929, residual 0.939; max 1.29 / 1.41 / 1.19), router **pass** (8.24% vs XLA 9.52%, −1.27 pp; no layer above XLA), logits/perplexity pass, worst_pos fail (p17@41) — same verdict as B.

| rel-L2 vs golden | L0 | L1 | L2 | L8 | L16 | L24 | L32 | L40 | L47 |
|---|---|---|---|---|---|---|---|---|---|
| attention, E | 0.0026 | 0.0067 | 0.0112 | 0.0238 | 0.0537 | 0.0959 | 0.1009 | 0.0940 | 0.0285 |
| attention, B | 0.0028 | 0.0076 | 0.0122 | 0.0228 | 0.0525 | 0.0932 | 0.0978 | 0.0897 | 0.0273 |
| attention, XLA | 0.0028 | 0.0091 | 0.0150 | 0.0274 | 0.0610 | 0.1016 | 0.1078 | 0.1001 | 0.0288 |
| residual, E | 0.0051 | 0.0076 | 0.0092 | 0.0166 | 0.0263 | 0.0418 | 0.0503 | 0.0503 | 0.0508 |
| residual, B | 0.0052 | 0.0077 | 0.0092 | 0.0163 | 0.0261 | 0.0408 | 0.0479 | 0.0483 | 0.0500 |

- **Measured (see T8 below for timing):** the megakernel's attention stage is at least as accurate as XLA attention where it is first applied (L0 attention 0.0026 vs 0.0028, same input); with lib MoE in both, E's residual error tracks B's within 0–5% at every depth, and both stay below XLA's.

## 2026-11-18 01:46–10:37 ET — T8 interleaved B → C → E benchmark, seq 768 and 2048 (Measured; E conditional on its validator verdict)

```bash
bash tools/chain_q2.sh E_cap E_stats warmup b768 b2048 b4096   # warmup compiled E at 2048 (1,385 s) and 4096 (1,543 s), runs not counted
python tools/bench_stats.py                                     # -> results/phase1/bench_bce.json
```
Each run is a separate process (`run_job.sh … --skip-compile`, shard-on-load, `benchmark_sampling` as in Phase 0); rounds interleaved B, C, E; same session. Builds: B = Phase 0 lib MoE (`traced_xlamoe_…_s768_greedy`, sweep builds at 2048/4096), C = Phase 0 megakernel (`traced_nki_…`), E = `traced_E_trn2_229_s{768,2048,4096}`. TKG latency in ms.

| Seq | n | B p50 | C p50 | E p50 | B p99 | C p99 | E p99 | C lead over B | attention B→E | fusion E→C | pooled std (C, E) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 768 | 5 | 10.493 ± 0.082 (10.447–10.639) | 10.291 ± 0.023 (10.269–10.324) | 12.141 ± 0.006 (12.133–12.150) | 11.229 | 10.757 | 12.563 | 0.202 | **−1.648** | **+1.850** | 0.017 |
| 2048 | 3 | 10.962 ± 0.070 (10.891–11.030) | 10.372 ± 0.096 (10.315–10.482) | 12.327 ± 0.068 (12.253–12.386) | 11.462 | 10.774 | 12.832 | 0.590 | **−1.365** | **+1.955** | 0.083 |
| 4096 | 3 | 11.116 ± 0.042 (11.085–11.164) | 10.397 ± 0.009 (10.388–10.406) | 12.564 ± 0.008 (12.557–12.573) | 11.677 | 10.800 | 13.107 | 0.719 | **−1.448** | **+2.167** | 0.009 |

(4096 added 17:06 ET; columns "attention B→E" and "fusion E→C" are arithmetic splits only — see Kevin's correction below: fusion's net gain is the "C lead over B" column, 0.20 / 0.59 / 0.72 ms.)

- **Measured:** E (lib MoE + the megakernel's attention stage as 48 separate NKI calls per step) is **1.4–1.6 ms/step slower than B** — close to Phase 0's config D (NxDI's NKI attention block, 12.08 ms at 768 and at 2048, Nov 12): +0.06 ms at 768, +0.25 ms at 2048 (different days). Unfused NKI attention, whichever kernel, costs ~1.5 ms/step on trn2.
- **Measured:** E's context slope is smaller than B's (768 → 2048: E +0.19, B +0.47, C +0.08 ms), so the NKI attention kernel does remove most of the XLA context cost; C additionally removes E's ~1.9 ms fixed cost.
- ~~**Decision rule (T8):** |C − E| = 1.85 ms (768) and 1.96 ms (2048) vs 2 × pooled std = 0.03 / 0.17 ms → fusion's share of C's lead is ~1.9 ms/step, and the Phase 4 fusion-depth sweep is back in scope.~~ **Corrected by Kevin, 2026-11-18:** the rule assumed E would land near B. E is 1.65 ms slower than B — like config D with a different attention kernel — so C − E is not fusion's share; it measures the cost of running NKI attention unfused, which fusion avoids. **Fusion's net gain is C vs B: 0.20 ms at 768, 0.59 ms at 2048.** The fusion-depth sweep is **not** back in scope. Next: T8b (per-call boundary cost; E with C's compiler flags).
- **Hypothesis (to be split by the T9 profile):** E's 1.9 ms is per-call cost that fusion removes, made of (a) per-layer rebuilds of layer-invariant work that C hoists out of its layer loop (attention mask and permuted cos/sin HBM scratch, built in every E call), (b) kernel-call boundaries that stop XLA from overlapping the next layer's weight DMA with the previous layer's compute, (c) compiler flags (E uses B's TKG −O1, cc-pipeline-tiling 2; C −O3, tiling 4, DMA vectorization). Only (b) is fusion in the narrow sense; (a) is cheap to hoist without fusing layers. The "fusion share" above is therefore an upper bound.
- Day-to-day: B at 768 is 0.18 ms slower than on Nov 12 (10.31 ± 0.01, pre-sharded weights; today shard-on-load) and C 0.14 ms slower; same-session differences are what count.
