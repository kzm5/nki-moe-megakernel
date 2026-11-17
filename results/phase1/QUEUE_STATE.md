# Q1+Q2 queue state (source of truth; queue: `docs/queues/Q1Q2.md`, read-only)

Updated: 2026-11-17 11:30 ET. Deadlines: Q1 Thu Nov 19 12:00 ET, Q2 Sun Nov 22 12:00 ET.

## Environment
- Worktree `~/nki-moe-megakernel/.claude/worktrees/q1q2` (branch `worktree-q1q2`), push `git push mine HEAD:phase0/trn2-setup` after `git pull --rebase mine phase0/trn2-setup`.
- Device/CPU jobs: `~/runs/run_py.sh NAME script args…` (cds to this worktree; log `~/runs/NAME.log`). Ledger `~/runs/ledger.jsonl` → `results/phase1/ledger.jsonl`.
- Phase-1 data: `~/runs/p1/` (`refs/`, `golden/`, `val/<run>/`, `src_wikitext/`).

## Tasks
| Task | Status | Commands / paths | Next step |
|---|---|---|---|
| Start of session | done 11:30 | CLAUDE.md updated; FINDINGS "State on start" | – |
| T1 restore | **done** 11:45: XLA bit-identical 24/24; refs `~/runs/p1/refs` (64) | `p1_xla_plain` | – |
| T2 validator | written, not frozen | `tools/validator/{validate,device,golden,mutants,corpus}.py`, `results/phase1/spec.json` | freeze after T3 golden |
| T3 corpus | corpus built (64 prompts, 16×512 ppl) | `tools/validator/corpus.py` → `results/phase1/corpus/corpus.json` | XLA refs (job 1), golden |
| T4 error budget | – | | |
| T5 mutants | code written | `tools/validator/mutants.py` | compile + validate m1–m4 |
| T6–T11 | – | | |
