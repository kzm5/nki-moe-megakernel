# Q1+Q2 queue state (source of truth; queue: `docs/queues/Q1Q2.md`, read-only)

Updated: 2026-11-18 23:30 ET. **Q1 and Q2 done** (deadlines Thu Nov 19 12:00 / Sun Nov 22 12:00). Summary: `SUMMARY_Q1Q2.md`. Device idle.

## Environment
- Worktree `~/nki-moe-megakernel/.claude/worktrees/q1q2` (branch `worktree-q1q2`); before every push `git pull --no-rebase mine phase0/trn2-setup`, then `git push mine HEAD:phase0/trn2-setup` (rule changed Nov 18; merges fine).
- `*.json` is gitignored: add result JSONs with `git add -f`.
- Jobs: `~/runs/run_py.sh NAME script args…` (cds to this worktree), `~/runs/run_job.sh` (benchmarks), `~/runs/prof/run_prof.sh` (profiles; kind E added). Ledger `~/runs/ledger.jsonl` → `results/phase1/ledger.jsonl`.
- Frozen validator: `results/phase1/FROZEN.md` (spec v1.0 + v1.1; default 1.1), guard hook in `~/.claude/settings.json`. Deleting anything: show Kevin a `du` list first.
- Status lines: `results/phase1/STATUS.md` (GitHub issues disabled on the repo).

## Tasks
| Task | Status | Paths |
|---|---|---|
| Start of session | done Nov 17 11:30 | FINDINGS "State on start" |
| T1 restore | done: XLA bit-identical 24/24 | `~/runs/p1/refs` (64 refs) |
| T2 validator | frozen v1.0 Nov 17 14:06; v1.1 Nov 18 14:06 | `tools/validator/`, `spec.json`, `spec_v1.1.json`, `FROZEN.md` |
| T3 corpus + golden | done | `results/phase1/corpus/corpus.json`, `~/runs/p1/golden` |
| T4 XLA error budget | done | `results/phase1/error_budget/xla.{json,csv}` |
| T5 mutants, Gate 1 | done: met (v1.0 and v1.1) | `results/phase1/reports_v10/`, `reports_v11/` |
| T6 E | done (first compile) | `megakernels/qwen3_moe/{attn_e_nki,qwen_attn_e}.py` |
| T7 validator on E | done: passes v1.1 | `reports_v11/E_*.json` |
| T8 B/C/E | done (768/2048/4096) | `results/phase1/bench_bce.json` |
| T8b null-kernel / C-flags | done | `results/phase1/bench_t8b.json`; `null_nki.py`, `qwen_null_nki.py`, `qwen_attn_e_cflags.py` |
| T9 profile E | done | `results/phase1/E_s768.json`, `~/runs/prof/p1_prof_E_s768` (983 MB raw) |
| T10 trn3 script | written, not run | `scripts/trn3_bringup.{sh,md}`, `tools/bench_only.py` |
| T11 skill audit | done | `results/phase1/SKILL_AUDIT.md`, `tests/skill_audit/` |
| H2.6 | done | `results/phase1/error_budget/xla_B_E.{json,csv}` |
| H2.2 | not started | – |
| Wrap-up | SUMMARY_Q1Q2 written | – |

## Builds (`~/qwen-30b-a3b/`)
XLA `traced_xla_trn2_229_s768_greedy` (production), `…_vcap` (validator capture), `…_vcap_m1` (m1); lib MoE `traced_xlamoe_trn2_229_s768_greedy`, `…_vcap`; megakernel `traced_nki_trn2_229_s768_greedy`; E `traced_E_trn2_229_s{768,2048,4096}`, `…_s768_vcap`; N `traced_Bnull_trn2_229_s768`; F `traced_Ecflags_trn2_229_s768`; sweep builds 256–4096; config D 768/2048. Validator outputs `~/runs/p1/val/<build>/`.
