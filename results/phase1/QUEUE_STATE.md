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
| T5 mutants + Gate 1 | chain running since 14:07 ET: xla_self → m1 → m2 → m4 → m3 → libmoe → nki | `chain_q1.sh` | then one golden `--stats` pass for all, `validate.py score` each |
| T6 E | code drafted (`megakernels/qwen3_moe/{attn_e_nki,qwen_attn_e}.py`), not compiled | | compile after chain |
| T10 trn3 script | drafted | `scripts/trn3_bringup.{sh,md}`, `tools/bench_only.py` | review in Q2 |
| T7–T9, T11 | – | | |
