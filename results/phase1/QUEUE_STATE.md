# Q1+Q2 queue state (source of truth; queue: `docs/queues/Q1Q2.md`, read-only)

Updated: 2026-11-20 10:52 ET. **Q1, Q2 and the early-finish items (H2.6, H2.2) done** (deadlines Thu Nov 19 12:00 / Sun Nov 22 12:00). Summary: `SUMMARY_Q1Q2.md`. Device idle, no jobs running; session stopped awaiting Q3 (kernel-edit approval pending).

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

Updated: 2026-11-23 16:10 ET. **Q3 done** (all tasks; `SUMMARY_Q3.md`), ahead of the Fri Nov 27 12:00 ET deadline. Device idle, no jobs running. Resumed Nov 22 after the Nov 20 21:40 ET – Nov 22 credit outage.

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
| T4 agent study A1/C1/A2/C2/C2r | **done**. Best eligible (harness n=2): A1 10.101, C1 **10.023** (iter 2), A2 10.060, C2 partial 10.035 (stopped by the usage limit, kept as record), C2r 10.082 (iter 2 9.859 non-deterministic). Final bench (Nov 23, V = C1 iter 2 `35309c3`, cross-process bit-identical): V 10.050 / 10.064 / 10.139 ms vs C 10.224 / 10.323 / 10.401 vs B 10.500 / 10.893 / 11.089 at 768 / 2048 / 4096 → **gain over B 4.28 / 7.61 / 8.57%; Gate 4 (5%) not met at 768** (2048/4096 builds not validatable) | `tools/agent_eval.sh`, `tools/bench_q3_final.sh`, `results/phase1/t4/`, `bench_q3_final.json`, branches `q3/t4-{base,a1,c1,a2,c2,c2r}` |
| Mid summary | written Nov 19 (T1–T3 complete) | `SUMMARY_Q3_mid.md` |
| Wrap-up | done Nov 23 16:10 ET | `SUMMARY_Q3.md`; 19 superseded agent builds / scoring dirs deleted (51 GB, approved Nov 23; `deletions.log`) |

## Disk and loading (Kevin, Nov 19 13:20 ET)
- Filesystem grown to 678 GB (223 GB free); keep the per-instruction JSONs (deletion request withdrawn).
- Root EBS stays at 3,000 IOPS: avoid swap. Pre-shard C (`traced_nki_trn2_229_s768_greedy`, the build loaded most in Q3) with `~/runs/val/preshard.py` after the T3 bench; load pre-sharded builds with `--save-sharded-checkpoint --skip-sharding` (the first makes NxDI read `weights/`). The frozen `device.py` cannot pass these flags, so validator loads still shard on load (page cache pre-warmed first).
- Kevin (Nov 20 ~00:30 ET): set `vm.swappiness=10` himself (runtime only); approved `MK_LOAD_PRESHARDED=1` in `main.py` (`a59ace1`, default off: with `--skip-compile` and an existing `weights/` it sets `save_sharded_checkpoint` / `skip_sharding`). Before any scored use: frozen validator on C's build with the hook and sharding on load, outputs bit-identical 64/64 (`tools/check_preshard_hook.sh`, runs after A1's 09:42 UTC deadline so it takes no device time from A1). If it passes, merge the hook into C1's worktree before launch; A1 is not restarted. The hook is a no-op for fresh agent builds (no `weights/`); pre-sharding each agent build (59 GB, ~990 s) would cost more than its one validator load.
- Baseline for the next validator load: Eh's load took 2,537 s (42 min; the 57 min reported earlier included the 888 s compile).

## T4 rules in force
- Non-deterministic variants (run-to-run ≠ 64/64) are reported separately, never as the best validated result (Kevin, Nov 20).
- A runtime-forced early hand-back is resumed with a neutral message and the deadline is extended by the idle time; active and wall time are reported for every run (Kevin, Nov 20).
- Disk 99 GB free on Nov 20 10:50 ET (validator dirs ~4.2 GB each, 112 GB total; compile cache 72 GB, active). Enough for A2/C2; pre-sharding the final-bench candidates (59 GB each) would need deletions — not planned.

# Q4Q5 queue state (queue: `docs/queues/Q4Q5.md`, read-only; Nov 24–29)

Updated: 2026-11-28 03:40 ET. **Q4, Q4b and Q5 done; results frozen; ready for the final AMI.** **Q4 done** (T4 at 4096 not run: needs a deletion). Deletions approved and done Nov 26 (52.3 GB). Q4b added by Kevin (Nov 24, `1bead66`): runs before Q5 (Q5 starts by Fri Nov 27 12:00). T6 stopped at its first load (aborted, no data). Deadlines (ET): Q4 Thu Nov 26 12:00; Q5 Sun Nov 29 18:00. trn3 day shelved (Kevin, Nov 23; queue `f6176a8`): no T1 merge deadline, no trn3 work.

## Environment
- Results in the main checkout (`phase0/trn2-setup`); kernel work in worktree `.claude/worktrees/q4/t1` (branch `q4/t1-merge-v`). Frozen validator still run from the q1q2 worktree.

## Tasks
| Task | Status | Paths |
|---|---|---|
| Start of session | done Nov 23 17:50 | FINDINGS "Q4 state on start"; CLAUDE.md reading list → Q4Q5 |
| T1 prepare V for merge | done Nov 23 18:45: NEFF not byte-identical (HLO hashes changed); outputs bit-identical 64/64 (7,264 positions, 16/16 ppl); **merged `a62b8af`** | branch `q4/t1-merge-v` (`3843abb`), `tools/q4_t1.sh`, `t4/V_t1_vs_c1_2.json` |
| T2 Eh at 2048/4096 | done Nov 24 16:37: Eh − C +0.399 / +0.357 ms (> 2σ), Eh − B −0.199 / −0.363 at 2048 / 4096 → context-growing gain is attention (Eh gets 33% / 50% of C − B); fusion side ~constant 0.35–0.40 ms | `bench_q4_t2.json`, FINDINGS T2 |
| T3 V determinism (3 captures) | done Nov 23 19:55: captures 3–5 bit-identical to captures 1 and 2 (7,264/7,264 each); V eligible | `results/phase1/t4/c1_2_r2r{3,4,5}*.json` |
| (chain) | done Nov 24 16:37 (`CHAIN_DONE`) | `tools/q4_chain.sh`, log `~/runs/p1/q4_chain.log` |
| T4 validate 2048/4096 builds | done: 2048 Nov 24, 4096 Nov 27 00:55 — XLA, B, C, V all PASS v1.1 at both (positions < 768 only) | `results/phase1/t4_s2048/`, `t4_s4096/` |
| T5 H2.4 all-reduce precision | done Nov 23 20:30: emulated bf16-per-add AR +43% KL vs fp32 sum; on device C and V bit-identical on 62/64 prompts (KL −0.2%) | `tools/h24_allreduce.py`, `h24.json`, `h24_C_vs_V.json`, FINDINGS T5 |
| T9 long-context corpus (Q4b, 10 h box) | done Nov 24 18:10 (1 h 15 min): XLA, B, C, V all pass v1.1 rows on the supplementary corpus at 2048 (positions 1,400–1,958); C = V on 16/16 prompts | `corpus_long/`, `FROZEN_long.md`, `tools/{corpus_long,validate_long}.py`, `tools/q4b_t9.sh` |
| T10 8192 sweep (Q4b, 10 h box) | done Nov 26 21:47: B 11.398, V 10.313 ms (n=2; V − B -9.52%: lead keeps growing); C compiles but hangs at runtime (wedged the device; driver reload); Eh not attempted | `bench_q4b_t10.json`, `tools/q4b_t10*.sh`, FINDINGS T10 |
| T11 reproducibility package (Q4b, 6 h box) | done Nov 26 22:13: proof from clean worktrees — validate B@768 PASS, bit-identical 64/64 to the original capture; bench 10.701 vs ledger 10.524 ± 0.088 | `scripts/reproduce/`, `docs/REPRODUCE.md`, FINDINGS T11 |
| T6 final interleaved session | done Nov 28 03:01: 54/54 runs; 768 V 10.052 vs B 10.522 (4.47%, Gate 4 not met) | `bench_q5_t6.json`, FINDINGS T6 |
| T7 tables and figures | done Nov 28 | `tools/make_figures.py`, `results/final/TABLES.md` + fig/tab CSVs |
| T8 freeze | done Nov 28 | `results/final/{SUMMARY_FINAL,NUMBERS}.md`, `tools/make_numbers.py`, RESUME, `~/runs/transcripts/` |
