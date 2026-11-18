# Q1+Q2 queue state (source of truth; queue: `docs/queues/Q1Q2.md`, read-only)

Updated: 2026-11-17 14:15 ET. Deadlines: Q1 Thu Nov 19 12:00 ET, Q2 Sun Nov 22 12:00 ET.

## Environment
- Worktree `~/nki-moe-megakernel/.claude/worktrees/q1q2` (branch `worktree-q1q2`), push `git push mine HEAD:phase0/trn2-setup` after `git pull --rebase mine phase0/trn2-setup`.
- Device/CPU jobs: `~/runs/run_py.sh NAME script args…` (cds to this worktree; log `~/runs/NAME.log`). Ledger `~/runs/ledger.jsonl` → `results/phase1/ledger.jsonl`.
- Phase-1 data: `~/runs/p1/` (`refs/`, `golden/` frozen; `val/<run>/` device outputs + `stats.pt`; `src_wikitext/`).
- Frozen validator: `results/phase1/FROZEN.md`; guard hook in `~/.claude/settings.json` (denies edits of frozen paths; use Edit tool / `git commit -F`).
- Device chain: `~/runs/p1/chain_q1.sh JOB…` (jobs xla_self m1 m2 m3 m4 libmoe nki libmoe_plain), log `~/runs/p1/chain_q1.log`.
- Status lines: `results/phase1/STATUS.md` (GitHub issues disabled on the repo).

## Tasks
| Task | Status | Commands / paths | Next step |
|---|---|---|---|
| Start of session | done 11:30 | CLAUDE.md updated; FINDINGS "State on start" | – |
| T1 restore | **done** 11:45: XLA bit-identical 24/24; refs `~/runs/p1/refs` (64) | `p1_xla_plain` | – |
| T2 validator | **frozen** 14:06 ET (19:06 UTC); hook verified | `tools/validator/`, `spec.json`, `FROZEN.md` | – |
| T3 corpus + golden | **done**: 64 prompts, 16×512 ppl, golden 81 files | `~/runs/p1/golden` | – |
| T4 error budget | **done** (XLA, all buckets) | `results/phase1/error_budget/xla.{json,csv}` | add B/E for H2.6 later |
| T5 mutants + Gate 1 | **done** 21:20 ET: Gate 1 met; B fails worst_pos (1 position), C passes | reports `~/runs/p1/reports/*.json`; `summarize_reports.py`; FINDINGS "Gate 1" | Kevin: decide on candidate worst-pos row |
| T6 E | first compile + validator run started 21:15 ET (time-box 10 h → 07:15 ET) | `p1_E_plain`: `device.py E ~/qwen-30b-a3b/traced_E_trn2_229_s768 ~/runs/p1/val/E_plain` | if OK: E capture build + 2048/4096 builds |
| Spec v1.1 (Kevin, Nov 18) | **re-frozen** 14:06 ET (v1.0 kept); validator `--spec {1.0,1.1}` default 1.1 | `spec_v1.1.json`, `FROZEN.md`, `FROZEN_v1.md` | rescore all builds under both after T8 4096 → `~/runs/p1/reports_v1/`, `reports_v11/` |
| T8 | 768 + 2048 done; 4096 running (chain_q2) | `tools/bench_bce.sh`, `tools/bench_stats.py` → `results/phase1/bench_bce.json` | correction recorded: fusion net gain = C vs B; sweep not in scope |
| T9 profile E @768 | after T8 4096 | `~/runs/prof/run_prof.sh p1_prof_E_s768 E ~/qwen-30b-a3b/traced_E_trn2_229_s768 768` | Table-5 format vs Phase 0 libmoe/nki tables |
| T8b (Kevin, Nov 18) | code written: N = B + null NKI kernel per layer (`null_nki.py`, `qwen_null_nki.py`), F = E with C's flags (`qwen_attn_e_cflags.py`) | `tools/bench_t8b.sh warmup; rounds 3` (B,C,E,N,F at 768) | after T9 |
| T10 trn3 script | drafted | `scripts/trn3_bringup.{sh,md}`, `tools/bench_only.py` | review in Q2 |
| T7–T9, T11 | – | | |
