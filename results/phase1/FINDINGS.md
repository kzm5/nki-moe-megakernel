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
