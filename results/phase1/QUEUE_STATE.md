# Q1+Q2 queue state (source of truth; queue: `docs/queues/Q1Q2.md`, read-only)

Updated: 2026-11-19 22:45 ET. **Q1, Q2 and the early-finish items (H2.6, H2.2) done** (deadlines Thu Nov 19 12:00 / Sun Nov 22 12:00). Summary: `SUMMARY_Q1Q2.md`. Device idle, no jobs running; session stopped awaiting Q3 (kernel-edit approval pending).

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
| T9 profile E | done | `results/phase1/E_s768.json`, `~/runs/prof/p1_prof_E_s768` (983 MB raw NTFFs; **keep — needed by Q3 T3**, deletion declined by Kevin Nov 19) |
| T10 trn3 script | written, not run | `scripts/trn3_bringup.{sh,md}`, `tools/bench_only.py` |
| T11 skill audit | done | `results/phase1/SKILL_AUDIT.md`, `tests/skill_audit/` |
| H2.6 | done | `results/phase1/error_budget/xla_B_E.{json,csv}` |
| H2.2 | done Nov 19 00:10 ET | `tools/h22_accum.py`, `results/phase1/h22.json` |
| Wrap-up | SUMMARY_Q1Q2 written | – |

## Builds (`~/qwen-30b-a3b/`)
XLA `traced_xla_trn2_229_s768_greedy` (production), `…_vcap` (validator capture), `…_vcap_m1` (m1); lib MoE `traced_xlamoe_trn2_229_s768_greedy`, `…_vcap`; megakernel `traced_nki_trn2_229_s768_greedy`; E `traced_E_trn2_229_s{768,2048,4096}`, `…_s768_vcap`; N `traced_Bnull_trn2_229_s768`; F `traced_Ecflags_trn2_229_s768`; sweep builds 256–4096; config D 768/2048. Validator outputs `~/runs/p1/val/<build>/`.

# Q3 queue state (queue: `docs/queues/Q3.md`, read-only; Nov 19–27)

Updated: 2026-11-19 22:45 ET. Checkpoint `SUMMARY_Q3_mid.md` due Sun Nov 22 12:00 ET; Q3 done Fri Nov 27 12:00 ET.

## Environment
- Results worktree `.claude/worktrees/q3` (branch `worktree-q3`; `git pull --no-rebase mine phase0/trn2-setup`, then `git push mine HEAD:phase0/trn2-setup`). Kernel work: worktrees on `q3/<task>` branches, pushed to `mine`, never merged here.
- Jobs: `MK_DIR=<worktree> ~/runs/run_py.sh …` / `run_job.sh`; frozen validator is run from the q1q2 worktree.

## Tasks
| Task | Status | Paths |
|---|---|---|
| Start of session | done Nov 19 10:37 | FINDINGS "Q3 state on start"; CLAUDE.md reading list → Q3 |
| T1 fix skills | done Nov 19 10:39 (`2af2cf5`, doc-only) | `.claude/skills/{nki-kernel-optimizer,neuron-profile}` |
| T2.1 fp32 router emulation | done Nov 19 10:58: fp32 router already in B/C/XLA; vs bf16 logits KL −21.6%, tie-tolerant flips +1.36 pp | `tools/router_emul.py`, `results/phase1/router_emul.json` |
| T2.2 fp32 router kernel | **skipped by rule** (flips do not fall ≥ 2 pp; the change would be a no-op) | FINDINGS T2.1 |
| T3.1 E vs C attribution | done Nov 19 11:10: per-call mask/RoPE build (software-DGE stall) ≈ 20 of E's +38.6 µs/layer | `tools/phase_attrib.py`, `results/phase1/t3/` |
| T3.2 E-copy changes | done Nov 19 20:20: Eh 10.707 vs E 12.255 / C 10.373 / B 10.579 ms (n=5) — 82% of E − C recovered; not within 2σ of C | branch `q3/t3-fusion` (`wt/t3`), `bench_q3_t3.json` |
| T4 agent study A1/C1(/A2/C2) | harness sanity PASS 10.311 ms (Nov 19 22:40); **A1 running** 03:42–09:42 UTC Nov 20 (`wt/a1`, branch `q3/t4-a1`); then C1 | `tools/agent_eval.sh`, `results/phase1/t4/`, `~/runs/agents/<run>/` |
| Mid summary | written Nov 19 (T1–T3 complete) | `SUMMARY_Q3_mid.md` |
| Wrap-up | due Nov 27 12:00 | |

## Disk and loading (Kevin, Nov 19 13:20 ET)
- Filesystem grown to 678 GB (223 GB free); keep the per-instruction JSONs (deletion request withdrawn).
- Root EBS stays at 3,000 IOPS: avoid swap. Pre-shard C (`traced_nki_trn2_229_s768_greedy`, the build loaded most in Q3) with `~/runs/val/preshard.py` after the T3 bench; load pre-sharded builds with `--save-sharded-checkpoint --skip-sharding` (the first makes NxDI read `weights/`). The frozen `device.py` cannot pass these flags, so validator loads still shard on load (page cache pre-warmed first).
- Baseline for the next validator load: Eh's load took 2,537 s (42 min; the 57 min reported earlier included the 888 s compile).
