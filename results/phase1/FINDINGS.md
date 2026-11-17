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
