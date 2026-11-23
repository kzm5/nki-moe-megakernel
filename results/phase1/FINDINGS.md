# Phase 1 findings (lab notebook, Q1+Q2 onward)

Queue: [`docs/queues/Q1Q2.md`](../../docs/queues/Q1Q2.md). State: [`QUEUE_STATE.md`](QUEUE_STATE.md). Ledger: `~/runs/ledger.jsonl` → [`ledger.jsonl`](ledger.jsonl). Times ET unless marked UTC. Q3 (Nov 19–27, [`docs/queues/Q3.md`](../../docs/queues/Q3.md)): newest Q3 entries directly below.

## 2026-11-22 22:04 → 2026-11-23 15:53 ET — Q3 T4 final interleaved bench: V (C1 iter 2) vs C and B (Measured)

`tools/q3_final_chain.sh` → `tools/bench_q3_final.sh`; log `~/runs/p1/bench_q3_final.log`; → `results/phase1/bench_q3_final.json`. 33 runs (+ 2 V compiles), all `EXIT_STATUS=0`; separate processes, rounds interleaved B → C → V, page cache pre-warmed before each run; C at 768 loads its pre-sharded weights, all others shard on load.

- **V = C1 iter 2 (`35309c3`, `MK_AGENT_VARIANT=1`):** at 768 the build `agent_c1_2` passed v1.1 (in-process run-to-run 64/64) and a second teacher-forced capture in a separate process (frozen `device.py`, 03:04–03:11 UTC) is **bit-identical on 7,264/7,264 positions** (`results/phase1/t4/c1_2_r2r.json`). V at 2048 / 4096: compiled from the same pinned worktree (57 / 72 min jobs incl. warm-up load); not validated (validator is 768-only).

| Seq | n | B p50 | C p50 | V p50 | V − C (2 × pooled std) | C gain over B | **V gain over B** (2 × pooled std) | Gate 4 (≥ 5%) for V |
|---|---|---|---|---|---|---|---|---|
| 768 | 5 | 10.500 ± 0.013 | 10.224 ± 0.007 | **10.050 ± 0.015** | −0.174 (0.024) | 2.63% | **4.28%** (−0.450 ms; 0.028) | **not met** |
| 2048 | 3 | 10.893 ± 0.008 | 10.323 ± 0.003 | **10.064 ± 0.003** | −0.259 (0.007) | 5.23% | **7.61%** (−0.829 ms; 0.012) | met (build not validated) |
| 4096 | 3 | 11.089 ± 0.032 | 10.401 ± 0.034 | **10.139 ± 0.019** | −0.263 (0.055) | 6.21% | **8.57%** (−0.951 ms; 0.053) | met (build not validated) |

p99 (ms): B 11.242 / 11.334 / 11.583; C 10.722 / 10.726 / 10.873; V 10.523 / 10.457 / 10.463.

- **Measured:** V is faster than C at every length by far more than 2 × pooled std (−0.17 / −0.26 / −0.26 ms = −1.7 / −2.5 / −2.5%). The collectives change takes the megakernel's net gain over B from 2.6 / 5.2 / 6.2% to 4.3 / 7.6 / 8.6%.
- **Measured:** at the validated length (768), V's gain over B is 4.28% — below Gate 4's 5%. At 2048 and 4096 it is 7.6% and 8.6%, but those builds are the validated source at another `--seq-len`, not validated builds (the frozen `validate.py run` passes no `--seq-len`). Plan discrepancy: Gate 4 does not state a sequence length; at 2048/4096 even C alone clears 5% (5.2 / 6.2%; T8 gave 5.4 / 6.5%).
- Measured: V's latency is nearly flat in context (768 → 4096: +0.09 ms; C +0.18, B +0.59).
- Day-to-day: C at 768 is 10.224 here vs 10.373 (T3 session, Nov 19) and 10.291 (T8, Nov 18); B 10.500 vs 10.579 / 10.493. Same-session differences are what count.

## 2026-11-22 17:36 → 22:03 ET — Q3 T4 run C2r (setup C), done; final bench started (Measured)

Worktree `wt/c2r` from `689a988`, prompt `results/phase1/t4/prompt_C2r.md`. Records `results/phase1/t4/c2r_{1,2}/`, diff `results/phase1/t4/c2r.diff` (only `transformer_qwen3_moe_speculative.py`, +68/−20, gated by `MK_AGENT_VARIANT=1` and BxS == 1), branch `q3/t4-c2r` pushed. Verdicts and run-to-run read from the validator reports.

| Iter | Commit | Change (agent's description) | Compile | Validator v1.1 | Run-to-run | KL ratio | TKG p50 (n=2) |
|---|---|---|---|---|---|---|---|
| 1 | `e07beb4` | LNC half-swap after each AR on `gpsimd_dma` (SBUF→SBUF); residual add folded into the collective helper; MoE transpose and gather/rearrange copies dropped | OK | PASS | 64/64 | 0.961 | **10.082** |
| 2 | `b3cdb3d` | + TP all-reduce → SBUF all-gather + rank-order fp32 local sum | OK | PASS | **0/64 (non-deterministic)** | 1.005 | 9.859 |

- C2r: 2 iterations / 2 compiles / 0 compile failures; 2/2 pass; best eligible **10.082 ms** (iter 1); iter 2 (9.859) non-deterministic → reported separately, not eligible. First validated improvement 00:45 UTC = 2 h 09 min after start. Wall 22:35–03:03 UTC = 4 h 28 min (stopped early: a 3rd iteration, ~2–2.3 h each, could not finish by 04:36); idle 0; active 4 h 28 min; device time 4 h 20 min; 0 runtime-forced hand-backs; 0 guard-hook or other blocked calls (self-report; transcript has no "Blocked"/"denied"); 0 interventions.
- Cross-run information: the transcript references only `~/runs/agents/c2r/`; no reads of other runs' records or branches. `~/runs/agents/c2/` permissions restored (755) after the run.
- Agent's hypothesis for iter 2's nondeterminism (Hypothesis, untested): not a read-before-complete race in the trace; each partition row of the gather buffer is 64 B made of four 16 B per-rank slices, and concurrent sub-64 B writes into one SBUF line may leave a stale slice. Orchestrator note: the all-gather + local sum idea is now seen three times — C1 iter 1 (non-deterministic), C2 iter 1 (deterministic, cross-process bit-identical), C2r iter 2 (non-deterministic) — so its determinism depends on the buffer layout of each implementation, consistent with C1's iter 2 fix (dedicated per-stage gather buffers).
- Profile (agent, Measured): iter 1 188.3 µs/layer, iter 2 183.3; iter 1 has 96 Mesh all-reduces of 4 KB, median 13.4 µs; iter 2's all-gathers median 10.6 µs; the `gpsimd_dma` LNC swap ≈ 1.45 µs per collective.

**Eligible variants across all runs (pass v1.1 + 64/64 run-to-run; harness TKG p50, n=2):** C1 iter 2 `35309c3` **10.023** · C2 iter 2 `b092e45` 10.035 · A2 iter 4 `9767038` 10.060 · C2r iter 1 `e07beb4` 10.082 · A1 iter 3 `00b7b0b` 10.101. The spread is within the harness's n=2 noise (~±0.05 ms); C1 iter 2 is taken for the final bench by the rule "best eligible harness p50".

**Final bench (started 03:04 UTC Nov 23, `tools/q3_final_chain.sh`):** (1) second teacher-forced capture of C1 iter 2 with the frozen `device.py` (cross-process run-to-run, `tools/r2r_rerun.sh c1 2`, pinned worktree `wt/c1_35309c3`); (2) V = C1 iter 2 compiled at 2048 and 4096 (`traced_V_q3final_s*`); (3) `tools/bench_q3_final.sh` interleaved B, C, V: 768 n=5, 2048 n=3, 4096 n=3. Caveat: the frozen validator runs at 768 only (`validate.py run` passes no `--seq-len`), so V at 2048/4096 is the same source and env as the validated 768 build, not a validated build — as for B and C's sweep builds in T8.

## 2026-11-22 17:36 ET — Q3 T4 run C2r (setup C restart) launched

- Kevin (Nov 22): the credit outage is a tooling failure, not a setup effect. C2 is kept as a partial record; C2r restarts it from scratch.
- Worktree `wt/c2r`, branch `q3/t4-c2r` from `689a988` (= C1's and C2's start). Prompt `results/phase1/t4/prompt_C2r.md` = C1's prompt with only run ids, paths and budget changed (checked by diff), waiting rule included. Fresh general-purpose subagent, launched 22:36 UTC; budget 22:36 UTC Nov 22 → **04:36 UTC Nov 23**.
- Environment difference from C2 (orchestrator, no hint): `~/runs/agents/c2/` is `chmod 000` for the run so C2r cannot read its predecessor's feedback; restored afterwards. Other runs' records stay as readable as they were for C2.

## 2026-11-22 17:30 ET — Q3 resumed after the credit outage (Measured)

- Commit `073ccec` (q3 worktree; `phase0/trn2-setup` at `5fa00c6`, RESEARCH_PLAN rev 45). Free disk **56 GB** of 678 (99 GB on Nov 20); RAM 119 GiB available, swap 0 used.
- Running jobs: none (`pgrep -af python`: system daemons only; no `neuron`/`run_py` processes; device idle per `neuron-ls`).
- C2 state: worktree `wt/c2`, branch `q3/t4-c2` at `b092e45`, clean (agent scratch only), now pushed to `mine`. Two iterations finished: iter 1 `1e7ddd6` (all-gather + local fp32 sum) and iter 2 `b092e45` (+ next layer's QKV weights prefetched to SBUF ahead of the MoE collective). Records copied to `results/phase1/t4/c2_{1,2}/`, diff `results/phase1/t4/c2.diff` (+137/−32; `transformer_qwen3_moe_speculative.py` gated by `MK_AGENT_VARIANT=1`, two legacy attention files extended with optional kwargs, default path unchanged).
- C2 stopped at **23:29:25 UTC Nov 20** ("You've hit your 5-hour limit", transcript `agent-ae89750a…`), 2 h 20 min into its budget, while waiting on iter 2's benchmark; the harness finished iter 2 at 00:46 UTC. No iteration 3 was committed. The orchestrator session stopped ~02:40 UTC Nov 21.
- Frozen validator v1.1 (harness run, `frozen_ok`): **both iterations PASS**, KL ratio 0.9635, 0 gross, ppl 8.383, in-process run-to-run 64/64; TKG p50 (n=2) iter 1 10.041, iter 2 **10.035 ms**. Cross-process check: the iter 1 and iter 2 captures (separate processes and builds; iter 2 changes only weight prefetch) are bit-identical on 64/64 prompts (`tools/r2r_compare.py` → `results/phase1/t4/c2_1_vs_c2_2.json`), so both are deterministic, unlike C1's iter 1 with the same idea.

## 2026-11-20 10:50 → 16:08 ET — Q3 T4 run A2 (setup A), done; C2 started (Measured)

Worktree `wt/a2` from `689a988` (= C1's start), prompt identical to A1's except run ids and budget (15:50–21:50 UTC). Harness records `results/phase1/t4/a2_{1..4}/`, diff `results/phase1/t4/a2.diff` (only `transformer_qwen3_moe_speculative.py`, +106/−25, gated), branch `q3/t4-a2` pushed. Checked against the validator reports.

| Iter | Commit | Change (agent's description) | Compile | Validator v1.1 | Run-to-run | TKG p50 (n=2) |
|---|---|---|---|---|---|---|
| 1 | `b35fac0` | fold each residual add into the AR step; MoE AR on the MoE output's layout (drops 8 transposes); add the result into each residual half, own half overlapping the LNC `sendrecv` | **FAIL** (NKI: no nested function definitions) | – | – | – |
| 2 | `100b402` | iter 1 with the views inlined | 947 s | PASS | 64/64 | 10.267 |
| 3 | `b3fc07a` | + MoE residual add also writes the next layer's attention input (GpSimd) into a fresh residual buffer | 758 s | PASS | 64/64 | 10.235 |
| 4 | `9767038` | + LNC `sendrecv` after each AR on `dma_engine=gpsimd_dma` (16 B/partition, standard-DMA fallback) | 741 s | PASS | 64/64 | **10.060** |

- A2: 4 iterations / 4 compiles / 1 compile failure; 3/3 compiled builds pass, all deterministic; best eligible **10.060 ms** (iter 4); first validated improvement 17:40 UTC (iter 2, 10.267, within noise of C) — first clear one 21:07 UTC = 5 h 17 min after start; wall 15:50–21:07 (stopped early: a 5th iteration might not finish by 21:50), idle 0 (no runtime-forced hand-back), active 5 h 17 min; device time 5 h 15 min; 0 guard-hook blocks; 1 other blocked call (compound git read under worktree isolation); 0 interventions.
- Like A1, A2's main gain came from moving the per-AR LNC exchange to `gpsimd_dma` (iter 4, −0.18 ms); A2 reached it in its 4th iteration, A1 in its 1st.
- Harness note: iter 4's second benchmark load took 3,718 s (sharding 3,053 s) — the page cache is pre-warmed only at the start of each iteration, so later loads in a long iteration can run cold. Left unchanged for C2 so pair 2 runs on the same harness as A2.
- C2 launched 21:09 UTC (budget to 03:09 UTC Nov 21), worktree `wt/c2` from `689a988`, prompt identical to C1's except run ids and budget.
- **C2 runtime-forced hand-back (Measured, transcript):** `[handback-send-enforce]` at 21:36:47 UTC, hand-back 21:37:01 (28 min into the budget; iter 1 `1e7ddd6` still in its validator job). Resumed immediately with one neutral message (21:37:40; intervention 1): idle ~1 min, deadline extended to **03:11 UTC** (rounded up). It was forced to hand back again at ~21:38 right after resuming (it ended its turn to wait on a Monitor) and asked to be resumed at each DONE or to have the waits allowed. Intervention 2 (~21:39): resumed, deadline **03:12 UTC**, and the prompt's own waiting rule restated (repeat the foreground `timeout 590 … until [ -f DONE ]` loop rather than ending the turn); no technical content. It also asked the user to confirm that its env-gated kernel edits count as approved — covered by Kevin's Q3 kernel-edit approval (env-gated, `q3/<task>` branches); nothing changed.
- **Cross-run information (Measured, transcript):** at 21:08:42 C2 listed `~/runs/agents/*/` and read the harness feedback of `base/1` (the unmodified megakernel: validator rows, latency, profile table), then analysed `~/runs/agents/base/1/prof.json` (baseline per-instruction profile). At 21:08:47 it tried to print A1's, A2's and C1's feedback; worktree isolation refused the command, and it did not try again ("chose not to read other agents' results"). So it saw that runs a1/a2/c1 exist and the baseline profile, but no other agent's changes or results. Its iter 1 (all-gather + fixed-order local fp32 sum) is the same idea as C1's iter 1, found independently. **Study limitation:** `~/runs/agents/` is readable by every run; the baseline profile is setup-C-type information that the harness did not intend to provide, and nothing stopped an A run from reading it either.

## 2026-11-20 05:21 → 10:38 ET — Q3 T4 run C1 (setup C), done (Measured)

Harness records `results/phase1/t4/c1_{1,2,3}/`, diff `results/phase1/t4/c1.diff` (only `transformer_qwen3_moe_speculative.py`, +101/−18, gated by `MK_AGENT_VARIANT=1`), branch `q3/t4-c1` pushed. Verdicts and run-to-run checked from the validator reports, not the agent's account.

| Iter | Commit | Change (agent's description) | Compile | Validator v1.1 | Run-to-run | KL ratio | TKG p50 (n=2) |
|---|---|---|---|---|---|---|---|
| 1 | `65486dc` | per-layer SB2SB all-reduce → all-gather + local fp32 sum of 4 partials | 942 s | PASS | **0/64 (non-deterministic)** | 0.943 | 10.034 |
| 2 | `35309c3` | + LNC exchange of the raw gathered partials (sum overlaps the exchange); a dedicated all-gather source buffer per stage | 723 s | PASS | 64/64 | 0.960 | **10.023** |
| 3 | `ede5298` | + fp32 sum added straight into the residual (one rounding); one MoE-output layout copy at BxS=1 | 726 s | **FAIL** (worst_pos: gross 2, max KL 0.959) | 64/64 | 1.000 | 9.900 |

| Metric | A1 (setup A) | C1 (setup C) |
|---|---|---|
| Iterations / compiles / compile failures | 3 / 3 / 0 | 3 / 3 / 0 |
| Validator v1.1 passes | 3 / 3 | 2 / 3 (iter 3 fails worst_pos) |
| Non-deterministic passing builds | 0 | 1 (iter 1; reported separately, not eligible) |
| Best eligible (pass + 64/64 run-to-run) | 10.101 ms (iter 3, `00b7b0b`) | **10.023 ms (iter 2, `35309c3`)** |
| Fastest build overall | 10.101 ms | 9.900 ms (fails v1.1) |
| First validated improvement | 06:02 UTC = 2 h 20 min after start (iter 1, eligible) | 12:02 UTC (iter 1, non-deterministic); first eligible 13:53 UTC = 3 h 32 min wall, 2 h 37 min active |
| Device time (sum of harness windows) | 5 h 53 min | 4 h 59 min |
| Wall / idle / active | 6 h 00 / 0 / 5 h 57 min (ended 09:39) | 5 h 17 min (10:21–15:38) / 55.0 min / 4 h 22 min (stopped early: a 4th iteration could not finish by 17:16) |
| Guard-hook blocks on frozen paths | 0 | 1, false positive: a Monitor `until`-loop that named the validator's log and used `pgrep -f` (no write attempted) |
| Other blocked tool calls | 2 (sleep rule; Monitor under worktree isolation) | 3 (writes to `/tmp` under isolation; `git -C .`; chained `sleep`) |
| Orchestrator interventions | 0 | 2 (resume after the runtime-forced hand-back; deadline +55 min) |

- Hypothesis (to test in the final interleaved bench): C1's iter 2 is faster than A1's best (10.023 vs 10.101 ms) and than C (10.311 in the harness); the two runs used different mechanisms (A1: fuse the AR result, residual add and LNC swap via SBUF→SBUF `gpsimd_dma`; C1: AG + local sum), so they may compose.
- C1's own reading of iter 3: the single-rounding residual update changes numerics at one fragile position; its untested suggestion is to keep iter 3's copy removal with the base's two-step rounding (Hypothesis).
- Setup C's per-layer ρ report was not available: the megakernel has no per-layer capture (validator `op_boundary` / `router` rows not measurable), so C's extra feedback was the full validator rows + the per-phase profile table.

### C1 iter 1 nondeterminism classified (Kevin's rule; orchestrator analysis, no hints to C1) — verdict: race → invalid

(a) Second teacher-forced capture of the same build with the frozen `device.py` (`tools/r2r_rerun.sh c1 1`, worktree pinned at `65486dc`, Nov 20 15:39–15:47 UTC, load 104 s), compared with the validator's capture (`tools/r2r_compare.py` → `~/runs/p1/val/agent_c1_1_r2r.json`; self-test on a bit-identical pair: 0 differences):

| | C1 iter 1, run 1 vs run 2 | reference: base/1 vs c1/1 (two different AR algorithms) |
|---|---|---|
| positions with any logit difference | 6,626 / 7,264 (91%) | 5,965 (82%) |
| max-abs logit difference | 11.86 | 6.47 |
| mean KL between the two | **9.4e-4** (= 42% of XLA's KL vs golden, 2.26e-3) | 7.9e-4 |
| p99 / max KL between the two | 1.18e-2 / 0.133 | 1.04e-2 / 0.101 |
| top-1 changes | 61 | 52 |
| each run's mean KL vs golden | 2.13e-3 / 2.17e-3 | 2.17e-3 / 2.13e-3 |

(b) Diff: no barrier, semaphore or explicit dependency removed (neither version has one); the AR algorithm and order changed (`nccl.all_reduce` → `nccl.all_gather` + fixed rank-order fp32 sum + one bf16 rounding).

- **Measured:** run-to-run differences are as large as replacing the AR algorithm outright (bf16-level perturbation amplified through 48 layers); neither run is less accurate than the base vs golden; no gross outliers.
- **Measured:** the nondeterminism cannot be reduction order: the local sum is fixed-order, and an fp32 sum of 4 bf16 values is exact unless their exponents differ by > 16 bits, so even a varying order would almost never change the result. C1's iter 2 change (a dedicated all-gather buffer per stage) made the output 64/64 bit-identical.
- **Hypothesis (mechanism):** a cross-rank buffer-reuse hazard — a reused all-gather source/destination address overwritten while another rank's transfer is still in flight; no explicit sync was removed, but `nccl.all_reduce`'s internal ordering was lost.
- **Verdict: race → invalid** (not "benign reduction-order nondeterminism"): although the magnitudes are bf16-like, a fixed-order sum cannot produce reduction-order nondeterminism, and a buffer change cured it. Reported, never eligible; it is not a determinism-for-0.28 ms trade-off. C1's iter 2 (deterministic, 10.023 ms) supersedes it.

### Earlier notes from the C1 run

- C1 launched 10:21 UTC (budget to 16:21), worktree `wt/c1` = `7fd70dc` + the `main.py` load hook (`689a988`; no-op for fresh builds), same prompt as A1 except the setup block.
- Iter 1 (`65486dc`): per-layer SBUF all-reduces → SBUF all-gather (4 ranks) + local fp32 sum of the 4 partials, then the LNC exchange. Compile 942 s; **validator v1.1 PASS, but run-to-run 0/64 identical** (all baseline and A1 builds 64/64); KL ratio 0.943, worst position max KL 1.06 (top-1 unchanged), ppl 8.369; **TKG p50 10.034 ms** (10.011, 10.057).
- **Intervention 1 (orchestrator, 12:13 UTC):** the agent handed back at 11:17 UTC, 56 min into its 6 h budget, while its iteration 1 was still benchmarking, saying it "was told to hand back" (not by the orchestrator). It did not resume after iteration 1 finished (12:02), so it was sent a neutral message: budget runs to 16:21, iteration 1's feedback is complete, continue unchanged. No technical content.
- **Validator spec gap (for Kevin):** v1.1's `run_to_run` row is non-blocking by design ("documented", `validate.py:12`), so a non-deterministic build passes. Orchestrator rule for T4 (not a spec change): non-deterministic variants are reported but not eligible as the best validated collectives result.
- **Where "told to hand back" came from (Measured, transcript `subagents/agent-…c1.jsonl`):** not the prompt, harness or hooks — none contain it, and the delivered prompts (`results/phase1/t4/prompt_{A1,C1}.md`) differ only in worktree path, harness arguments, budget times and the SETUP block. At 11:17:35 UTC the Claude Code subagent runtime injected `[handback-send-enforce] Your report has not been delivered. Call SubagentHandback(...) now; the call ends your run.` right after C1 ended a turn with plain text ("Still waiting for iteration 1 benchmark (DONE monitor armed)") while its only waits were Monitor tasks; C1 complied at 11:17:56. It fired again at 12:16:41 after the resume; C1 ignored it the second time and kept working. A1 never received it (0 occurrences): A1 waited with `run_in_background` Bash loops (6 uses, 1 Monitor), which the runtime counts as the subagent's own background work; C1 used Monitor (5 uses). Cause: the runtime's hand-back enforcement does not treat a pending Monitor as background work. Not an agent decision; not a setup difference.
- **Deadline extension (Kevin, Nov 20):** C1 was idle from the hand-back (11:17:57) to the resume (12:12:59) = **55.0 min**; its deadline moved from 16:21 to **17:16 UTC** so A1 and C1 get the same active time. Sent to C1 as a neutral budget message (intervention 2).

| Run | Wall window (UTC) | Wall | Idle (handed back, not running) | Active | Orchestrator interventions |
|---|---|---|---|---|---|
| base/1 (harness sanity, no agent) | Nov 20 01:36–03:40 | 2 h 03 min | – | – | – |
| A1 | Nov 20 03:42–09:42 | 6 h 00 min | 0 | 6 h 00 min | 0 |
| C1 | Nov 20 10:21–17:16 | 6 h 55 min | 55.0 min | 6 h 00 min | 2 (resume 12:13; deadline +55 min 12:58) |

- **Nondeterminism of C1 iter 1 (`65486dc`), Kevin's classification (orchestrator analysis only; C1 gets no hints).** (b) Diff read: the change replaces `nccl.all_reduce` (in `_sb2sb_all_reduce_gather`) by `nccl.all_gather` into a new SBUF buffer + a fixed rank-order fp32 sum (r0+r1, +r2, +r3, one rounding to bf16), keeping the same LNC `sendrecv` (`pipe_id=0`). **No barrier, semaphore or explicit dependency was removed or relaxed** — neither version has an explicit sync; both rely on NKI's implicit tensor dependencies (collective → read → `sendrecv`). **The all-reduce algorithm and summation order did change.** A fixed-order local sum cannot by itself vary between runs, so the varying input must come from the all-gather (slot order by arrival, or reads of `ag_sb` before all remote writes land) — (a) decides which. (a) Run-to-run quantification: a second teacher-forced capture with the frozen `device.py` (`tools/r2r_rerun.sh c1 1`, pinned worktree `wt/c1_65486dc`) runs after C1's deadline so it takes no device time from C1; `tools/r2r_compare.py` self-test on a known bit-identical pair: 0 differences.
- The agent's own profile analysis of the baseline: each per-layer all-reduce ≈ 18 (attention) / 21.5 (MoE) µs launch-to-residual-add, of which ~3.7 µs is data transfer (16-byte pieces per partition); the rest is waiting (~5 µs before the first transfer, ~4 µs between phases) plus ~3–4 µs of post-collective copies. The phase table's `xla.cc` rows in the megakernel are the in-kernel MoE all-reduce (its launch instruction comes from `ops.py`).

## 2026-11-20 04:42 → 05:20 ET — pre-sharded-load hook checked: bit-identical, load 149 s vs 806 s (Measured)

Kevin approved (Nov 20) `vm.swappiness=10` (set by him, runtime) and `MK_LOAD_PRESHARDED=1` in `main.py` (`a59ace1`, default off). `tools/check_preshard_hook.sh`: frozen validator v1.1 on C's pre-sharded build, `MK_DIR=q3`, page cache pre-warmed, run after A1's deadline.

| Run | LOADED | Sharding on load | Job wall | Validator v1.1 |
|---|---|---|---|---|
| `MK_LOAD_PRESHARDED=1` (log: "Loading presharded checkpoints") | **149 s** | – | 10.1 min | PASS |
| sharding on load, swappiness 10 | 806 s | 537 s | 23.1 min | PASS |
| earlier: T4 base/1, swappiness 60 (fresh build) | 2,049 s | 1,407 s | | PASS |
| earlier: Eh, cold page cache | 2,537 s | | | PASS |

- **Measured: device outputs bit-identical with and without the hook — teacher-forced scores 64/64, perplexity passages 16/16.** The hook is cleared for scored runs. It is now in C1's worktree (cherry-pick `689a988` on `q3/t4-c1`) and set by `agent_eval.sh` for validator runs; it only acts on builds with `weights/` (C's), so fresh agent builds still shard on load. A1 ran without it (not restarted).
- Measured: swappiness 10 alone cut sharding on load from 1,407 s to 537 s.

## 2026-11-19 22:42 → 2026-11-20 04:39 ET — Q3 T4 run A1 (setup A: compile errors + validator PASS/FAIL + TKG p50) (Measured)

Fresh general-purpose subagent with the nki-kernel-optimizer skill, prompt `results/phase1/t4/prompt_template.md` (setup A), worktree `wt/a1` from `7fd70dc`, budget 6 h (03:42–09:42 UTC). Orchestrator gave no hints beyond the setup. Harness records: `results/phase1/t4/a1_{1,2,3}/`; diff vs start `results/phase1/t4/a1.diff`; branch `q3/t4-a1` pushed.

| Iter | Commit | Change (agent's description) | Compile | Validator v1.1 | TKG p50 (n=2) |
|---|---|---|---|---|---|
| 1 | `550ba57` | per all-reduce: each LNC core adds the AR result into its own H-half of `residual_sb`, then swaps halves directly into `residual_sb` via `sendrecv` on `gpsimd_dma` (SBUF→SBUF); drops the gather copy, the identity rearrange copy, the separate residual add and 8 MoE-output transpose copies (BxS=1) | 897 s | PASS | 10.128 |
| 2 | `3ff90a7` | + the MoE swap writes the next layer's attention input directly; `residual_sb` refreshed off the critical path | 703 s | PASS | 10.103 |
| 3 | `00b7b0b` | + residual swaps on their own `sendrecv` pipe (`pipe_id=1`) | 702 s | PASS | **10.101** |

| Metric (A1) | Value |
|---|---|
| Iterations / compiles | 3 / 3 (all compiled, all passed v1.1) |
| Device time | 5 h 55 min (harness wall 03:43–09:39 UTC; the device was held throughout) |
| Best validated latency | 10.101 ms (iter 3; harness n=2; unmodified megakernel in the harness 10.311, T3 session C 10.373 ± 0.094) |
| First validated improvement | iter 1, feedback at 06:02 UTC = 2 h 20 min after start |
| Blocked attempts on frozen paths | none (self-report; diff touches only `transformer_qwen3_moe_speculative.py`, +90/−32, gated by `MK_AGENT_VARIANT=1` and BxS == 1) |
| Other blocked tool calls | 2, harness policy, not validator-related: a chained `sleep` (sleep rule) and a Monitor loop (worktree isolation) |
| Human interventions | 0 |

- Hypothesis (to test in the final interleaved bench): A1's ~0.2 ms gain is real — it is ~2× the harness's run-to-run spread and the three iterations are monotone, but the comparison is with other sessions' C numbers.
- The agent argued the per-layer AR count cannot be reduced (RMSNorm and router/QKV need the full H after each AR; the two ARs per layer are dependent), and that the subkernels' other LNC exchanges cannot use `gpsimd_dma` (partition dim 1 at T=1).

## 2026-11-19 20:20 → 22:40 ET — C pre-sharded; T4 harness sanity iteration on the unmodified megakernel (Measured)

- **Pre-shard (Kevin's instruction, Nov 19):** `~/runs/val/preshard.py nki …/traced_nki_trn2_229_s768_greedy 768` wrote 4 shards (59 GB) into `weights/` in 990 s (job 17.3 min, peak RSS 121.6 GiB, swap peaked ~40 GB during it). C is now loaded with `--save-sharded-checkpoint --skip-sharding` by `bench_q3.sh` and `agent_eval.sh` (first flag makes NxDI read `weights/`). Logged in the ledger and `~/runs/deletions.log`.
- **Next validator load (requested comparison):** the sanity iteration's validator load took **2,049 s (34 min) vs Eh's 2,537 s (42 min)**; the "57 min" reported earlier was Eh's `LOADED` time including its 888 s compile. 1,407 s of the 2,049 s was sharding on load with swap up to ~28 GB — the frozen `device.py` cannot pass `--save-sharded-checkpoint`, so validator loads still shard on load, and the ~121 GiB sharding peak overflows RAM (swappiness 60). Changing swappiness to 10 was attempted and denied by the permission system; not done.
- **Harness sanity iteration (`tools/agent_eval.sh C base 1 wt/base`, commit `7fd70dc`, unmodified megakernel):** compile 929 s; **validator v1.1 PASS** (KL ratio 0.961, 0 gross, ppl 8.385, run-to-run bit-identical); **TKG p50 10.311 ms** (10.283, 10.340; C in the T3 session 10.373 ± 0.094); profile 192.7 µs/layer (same as T3.1); total 2 h 03 min. Found and fixed: `main.py --compile-only` is parsed but unused, so the separate compile step loaded and warmed the model (~19 min wasted); compile now happens inside the validator's `run` (one load instead of two). Note for setup C: in the megakernel, the phase table's `xla.*` rows are instructions without NKI source (collective glue), labelled by the nearest-anchor rule.

## 2026-11-19 11:15–12:45 — Q3 T3.2: Eh = E with the mask/RoPE build hoisted to once per step (Measured; timing pending)

Branch `q3/t3-fusion` (`a66879a`, worktree `.claude/worktrees/q3/wt/t3`, pushed): new kernel copy `megakernels/qwen3_moe/attn_e_hoist_nki.py` (`attn_e_hoist_tkg` = `attn_e_tkg` minus the two per-call builders; mask and permuted cos/sin are inputs) and, in `qwen_attn_e.py`, `ATTN_E_HOIST=1` (default 0 → unchanged E) builds the same tables in the graph once per traced step (cached on the `position_ids` tensor shared by all layers) with the builders' exact layouts.
- **Validator v1.1 (frozen, run from q1q2 with `MK_DIR=…/wt/t3 ATTN_E_HOIST=1`, kind E): PASS**, `frozen_ok`, KL ratio 0.9670, 0 gross, run-to-run bit-identical; **teacher-forced scores bit-identical to E on 64/64 prompts** (the hoist changes timing only). Build `traced_Eh_trn2_229_s768`: compile 888 s; load 3,425 s because the page cache was cold (root EBS capped at 3,000 IOPS → mmap faults at ~12 MB/s; a `dd` pre-read of the 57 GB checkpoint takes 81 s, now done before every device job).
- **Profile (same method as T3.1, `p1_prof_Eh_s768`, rank vnc_1, bucket 128): Eh 199.3 µs/layer vs E 231.3 and C 192.7 → the hoist recovers 32.0 of E's 38.6 µs/layer (83%)**; graph time 11.10 → 9.57 ms. Recovered: idle −16.0, QKV −6.7 (its DMA now 11.2 µs for 5.2 MB vs 28.2 µs for 3.8 MB: the slow QKV weight load in E was DMA-queue contention with the software-DGE builds, not missing prefetch — cause 2 of T3.1 is a consequence of cause 1), mask −4.3, RoPE −2.7, XLA glue −1.9, MoE gate_up −1.8, block_other −1.5.
- **Remaining Eh − C = +6.6 µs/layer:** X load / Y store +5.6 (inherent to an unfused kernel fed from the graph), library-vs-vendored MoE (other +4.5, router +2.7), partly offset by collectives (XLA CC −4.2) and C's in-kernel layer glue (−3.9). No second change was tested: cause 2 disappeared with cause 1, and the rest is the call boundary itself or the MoE kernel choice.
- `results/phase1/t3/Eh_s768_b128_phase.json`.

**Timing (Nov 19 12:42 → 20:17 ET; `tools/bench_q3.sh t3 5 B C E Eh`, interleaved separate processes, n=5 at 768, 20/20 EXIT 0, page cache pre-warmed before each run; → `results/phase1/bench_q3_t3.json`):**

| Config | p50 TKG (ms) | − E | − C (2 × pooled std) | − B (2 × pooled std) |
|---|---|---|---|---|
| B lib MoE | 10.579 ± 0.129 | −1.676 | +0.206 (0.225) | – |
| C megakernel | 10.373 ± 0.094 | −1.882 | – | −0.206 (0.225) |
| E | 12.255 ± 0.100 | – | +1.882 (0.194) | +1.676 (0.230) |
| **Eh (hoisted mask/RoPE)** | **10.707 ± 0.086** | **−1.547** | **+0.335 (0.180)** | **+0.128 (0.218)** |

- **Measured:** hoisting the mask/RoPE build recovers **1.547 of E's 1.882 ms/step over C (82%)** = 32.2 µs/layer, matching the profile's 32.0 µs/layer.
- **Measured:** Eh is still 0.335 ms/step above C (> 2 × pooled std 0.180), so the queue's condition "E within 2 × pooled std of C" is **not met**; the remainder (~7 µs/layer) is the call boundary's X load / Y store and the library-vs-vendored MoE kernel (profile above). Eh is within noise of B (+0.128 < 0.218).
- **Result (T3):** on trn2, ~82% of the cost fusion avoids for this attention stage is per-call setup — rebuilding the mask/RoPE tables with software-DGE DMAs, which also stalls the QKV weight load — and an unfused kernel avoids it too by taking the tables as inputs built once per step. What fusion still buys beyond that is ~7 µs/layer (0.34 ms/step at 768): no activation round trip through HBM per call, plus its own MoE kernel. Fusion's net gain over B in this session: 0.206 ms (T8: 0.20).
- Same-session drift vs T8 (Nov 18): B 10.58 vs 10.49, C 10.37 vs 10.29, E 12.26 vs 12.14.

## 2026-11-19 10:45–11:10 — Q3 T3.1: where E's +38.6 µs/layer goes, E vs C by kernel phase (Measured)

Method: `tools/phase_attrib.py` on the per-instruction JSON (`neuron-profile view --output-format json --ignore-event-trace --ignore-nc-buf-usage`) of the least-skewed rank (vnc_1), TKG bucket 128, seq 768. Each instruction is anchored to a phase by source file + enclosing function (attention_block_tkg / qkv_tkg / attention_tkg / output_projection_tkg / transformer_qwen3_moe_speculative builders / MoE kernels / XLA `module.py`); shared helpers, no-source instructions and DMA records take the phase of the nearest anchored instruction in time. Wall = fair-share sweep over all engine instruction intervals (sums to the step time); "idle" = no engine instruction running (split by whether DMA is in flight). E: the T9 NTFF (`~/runs/prof/p1_prof_E_s768`). **C: the Phase-0 NTFFs had been deleted (Nov 11), so C was re-profiled** with the same `profile_decode.py` (`~/runs/prof/run_prof.sh p1_prof_C_s768 nki …s768_greedy 768`, 2.3 min device, peak RSS 120 GiB). Totals reproduce T9: E 231.3, C 192.7 µs/layer (T9: 192.5). → `results/phase1/t3/{E,C}_s768_b128_phase.json`.

| Phase | E wall | C wall | E − C | E DMA µs / MB | C DMA µs / MB |
|---|---|---|---|---|---|
| idle (no engine busy) | 45.0 | 28.0 | **+17.1** | 28.4 with DMA in flight | 11.3 with DMA in flight |
| attn.qkv | 19.9 | 13.4 | **+6.5** | 28.2 / 3.81 | 12.7 / 4.97 |
| attn.x_load_y_store (E's X load, Y store) | 5.2 | 0.0 | +5.2 | 4.6 / 0.25 | – |
| xla.ops (graph glue) | 11.8 | 7.2 | +4.7 | 14.9 / 5.54 | 15.6 / 4.95 |
| attn.mask_build | 4.3 | 1.0 | **+3.4** | 3.6 / 0.37 | 0.2 / 0.02 |
| moe.other | 8.8 | 5.7 | +3.1 | 7.0 / 0.83 | 6.3 / 0.77 |
| attn.rope_build | 2.7 | 0.0 | **+2.7** | 3.7 / 0.40 | 0.4 / 0.04 |
| moe.router | 12.6 | 10.2 | +2.4 | 6.5 / 1.54 | 4.6 / 0.70 |
| attn.block_other | 5.9 | 3.9 | +1.9 | 19.7 / 1.22 | 4.8 / 0.56 |
| moe.cc | 1.8 | 0.0 | +1.8 | – | – |
| attn.o_proj / attn.cc / qk_rope | 6.6 / 4.6 / 1.1 | 6.0 / 4.0 / 1.0 | +0.6 / +0.5 / +0.1 | | |
| moe.gate_up / moe.down | 53.3 / 27.0 | 53.5 / 27.7 | −0.2 / −0.7 | 60.1 / 12.86, 34.9 / 5.33 | 60.9 / 13.38, 36.8 / 6.11 |
| attn.kv_update / qk_norm / allreduce_gather / core | 1.3 / 2.6 / 2.6 / 9.5 | 1.6 / 3.2 / 3.4 / 10.9 | −0.3 / −0.6 / −0.8 / −1.4 | | |
| xla.cc | 4.8 | 8.1 | −3.3 | – | – |
| mk.body (C's in-kernel layer glue) | 0.0 | 3.9 | −3.9 | – | 3.1 / 0.73 |
| **total** | **231.3** | **192.7** | **+38.6** | | |

- **Measured — cause 1, per-call mask/RoPE build (~20 µs/layer):** E rebuilds the attention mask and permuted cos/sin in HBM on every call (`_build_full_mask_hbm`, `_build_permuted_rope_hbm_from_pos`: +3.4 +2.7 µs of instructions). Their element-strided DMAs run on **software DGE** (`software_dynamic` queue Q2 on all 16 DMA engines, ~0.85 MB/layer moved in ~13.6 µs), and the attention block's first consumer, `_to_sbuf(cos/sin)` at `attention_block_tkg.py:859`, waits on them: gaps > 3 µs total 17.0 µs/layer in E, **15.5 µs of it ending at line 859** (preceded by `qkv_tkg.py:1488`, the QKV weight DMA); C has 1.3 µs/layer of such gaps. With block_other's +1.9 (its DMAs wait on the same loads), cause 1 ≈ 3.4 + 2.7 + 1.9 + ~13 ≈ 20 µs/layer ≈ 1.0 ms/step. C builds the tables once per step (1.0 µs/layer amortized).
- **Measured — cause 2, QKV weight load (+6.5 µs/layer):** E's QKV phase spends 28.2 µs DMA-active for 3.8 MB vs C's 12.7 µs for 5.0 MB; C streams the next layer's weights while the previous layer runs, E cannot prefetch across the call boundary.
- **Measured — cause 3, unfused glue (~+6 µs/layer):** E's X load / Y store (+5.2) and XLA glue ops (+4.7), against C's in-kernel layer body (3.9).
- **Measured — not attention (~+4 µs/layer):** E uses the library MoE kernel, C its vendored MoE: router +2.4, other +3.1, MoE CC +1.8 vs XLA CC −3.3.
- Hypothesis (T3.2 tests it): hoisting the mask/RoPE build to once per step recovers ~20 µs/layer (~1 ms/step at 768), about half of E − C.

## 2026-11-19 10:40–10:58 — Q3 T2.1: router precision emulated in a bf16 CPU model; T2.2 skipped by rule

`~/runs/run_py.sh p1_t2_router_emul tools/router_emul.py --out ~/runs/p1/t2/router_emul.json` (CPU, 15.6 min, 4.7 GiB) → `results/phase1/router_emul.json`. Whole model in bf16 (weights, activations, residual stream; HF layers, CPU AMX), only the router varied; scored against the frozen golden with the validator's own `kl` / `valid_top8` (tie-tolerant flip rule) over all 7,264 corpus positions × 48 layers, plus the 16 × 512 perplexity set.

| Variant | mean KL | top-1 = golden | ppl (golden 8.342) | router flips, validator rule | strict flips (set differs) |
|---|---|---|---|---|---|
| bf16logits (logits rounded to bf16, fp32 softmax/top-k) | 2.43e-3 | 98.49% | 8.342 | 9.53% | 13.87% |
| **fp32** (bf16 operands, fp32 logits, softmax, top-k) | **1.90e-3** | **98.82%** | 8.329 | 10.89% | 10.89% |
| fp32in (fp32 + router input = un-rounded fp32 RMSNorm output) | 2.05e-3 | 98.61% | 8.342 | 10.65% | 10.65% |

- **Measured (source + build configs): B, C and XLA already compute router logits and top-k in fp32.** Every build's `neuron_config.json` has `router_config.dtype = float32` (NxDI's Qwen3-MoE sets it "for accuracy"; ours copies it in `megakernels/qwen3_moe/qwen.py:269`). XLA: `RouterTopK` casts the bf16 hidden state to fp32 for an fp32 matmul (`--auto-cast=none`), float64 softmax, top-k on fp32 logits. B: `moe_block_tkg` copies the RMSNorm output to `router_mm_dtype`; `router_topk` accumulates in fp32 PSUM, keeps `router_logits_sb` fp32 and runs `max8` / `nc_find_index8` on it (only the HBM copy of the logits is bf16). C: the legacy `router_topk` does the same with bf16 weights (fp32 PSUM/SBUF logits). So the "fp32" row is the status quo of all three builds, not a new lever.
- **Measured:** against bf16 logits, fp32 logits cut mean KL by 21.6% and strict flips by 2.98 pp; on the validator's tie-tolerant rule flips *rise* 1.36 pp, because bf16 logits create exact ties at the 8th/9th expert that the rule forgives (the tie rule is lenient to low-precision logits).
- **Measured:** the remaining router-local lever (fp32in) does not help: −0.24 pp flips, +7.5% KL, ppl +0.013 vs fp32 (all within the variant-to-variant noise of this emulation).
- **Decision rule applied:** the validator-rule flip rate falls by less than 2 pp (it rises) → **skip T2.2; this emulation is the T2 result.** Independently, the T2.2 kernel change (fp32 router logits + top-k in B) would be a no-op. If Kevin wants strict flips used instead (2.98 pp ≥ 2 and KL −21.6% ≥ 10%), the rule would pass, but the fp32 router it would build already exists.
- Emulation fidelity (Measured): the fp32 row's KL (1.90e-3) is close to B's device KL (0.87 × XLA's 2.26e-3 = 1.97e-3); its flip rate (10.89%) is 1.4 pp above XLA's device 9.52% and ~2.8 pp above B's, so compare emulated variants with each other, not with device rows.
- Per-layer flips (fp32) peak at L24–33 (15.9 / 15.3%), L0 1.7%. Residual rel-L2 at the 512 sampled positions L0/L47: 0.0096/0.052 (bf16logits), 0.0060/0.046 (fp32), 0.0058/0.045 (fp32in).
- **Plan discrepancy (for Kevin):** RESEARCH_PLAN's Phase 3 / Q3 T2 treat fp32 router logits as an untested accuracy lever; on this stack they are already in every build, so router flips (~87% of MoE error, Phase 0 Task R) come from upstream bf16 hidden-state error, not router precision. **Hypothesis:** the accuracy levers left are upstream: fp32 residual stream or bf16-input rounding of the expert GEMMs (H2.2: 30% of XLA's KL).

## 2026-11-19 10:37–10:39 — Q3 T1: skill fixes (doc-only, commit `2af2cf5`)

- Re-verified on SDK 2.29 before editing (Measured): `nisa.dge_mode` members `hwdge, none, swdge, unknown`; `nki.jit(k, platform_target='trn2')` raises `ValueError … deprecated. Use NEURON_PLATFORM_TARGET_OVERRIDE`.
- nki-kernel-optimizer: `hw_dge` → `hwdge` (trn2-architecture.md ×2); `platform_target=` removed from quickref ×2, benchmarking-api ×2, benchmark.py docstrings ×3 (env var instead); venv → `~/venv-neuron-229/activate-neuron.sh` (SKILL.md, templates.md); `nki.simulate` bf16 caveat in SKILL.md Correctness; "trn3 only (NCC_IBIR530)" on `nc_matmul_mx` / `quantize_mx`; header says trn2.
- neuron-profile: JSON `timestamp`/`duration`/`evt_wait_time` in ns (not ps), `--show-trace` in ps; clock ~1.2 GHz measured; CC-core id is per rank (12–15 seen, not 8) in SKILL.md and layer_latency.md; new "Multiple ranks and LNC=2" section in reading_the_profiler.md (least-skewed rank, LNC=2 union/sum semantics).

## 2026-11-19 10:37 — Q3 state on start (Measured)

- Commit `5f2c1b4` (remote tip of `phase0/trn2-setup`); worktree `.claude/worktrees/q3` (branch `worktree-q3`, pushed as `HEAD:phase0/trn2-setup`); the frozen validator stays in `.claude/worktrees/q1q2` and runs from there.
- No device or CPU job running; `neuron-ls` idle (1 device, 4 logical cores, LNC=2). RAM 120 GiB available of 124.
- Disk: **42 GB free** of 484 GB (92% used). Kernel-variant compiles will need deletions; Kevin approves from a `du` list.
- Builds present (`~/qwen-30b-a3b/`): XLA s768 production/vcap/vcap_m1; B s768 production/vcap/router; C s768 + sweeps 256–4096; E s768/2048/4096 + s768_vcap; N; F; config D 768/2048. Profiles: E s768 (983 MB NTFFs), B/C/D s768, B/C s2048.
- H2.2 done (Nov 19 00:10 ET; `h22.json`): bf16 expert-GEMM inputs = 30% of XLA's KL, bf16 accumulation 44%; `tools/h22_accum.py` is the base for T2.1.

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

## 2026-11-18 17:45 → 22:50 ET — T8b: per-call boundary cost and compiler flags, seq 768 (Measured)

Added by Kevin (Nov 18). N = B + one null NKI kernel per layer on TKG (`megakernels/qwen3_moe/null_nki.py`: identity copy of the attention output HBM → SBUF → HBM with E's load/store layout, LNC=2 sharding and core barrier; `qwen_null_nki.py`; compile 1,002 s). F = E compiled with C's compiler flags (`qwen_attn_e_cflags.py` → `NeuronQwen3MoeForCausalLMV3.get_compiler_args`: TKG −O3, cc-pipeline-tiling 4, DMA vectorization; compile 871 s). `tools/bench_t8b.sh rounds 3`: interleaved B → C → E → N → F, separate processes, same session; 15 runs, all EXIT 0. → `results/phase1/bench_t8b.json`.

| Config | p50 TKG (ms) | range | p99 (ms) |
|---|---|---|---|
| B lib MoE | 10.521 ± 0.086 | 10.445–10.614 | 11.294 ± 0.144 |
| C megakernel | 10.358 ± 0.088 | 10.266–10.441 | 10.919 ± 0.229 |
| E (megakernel attention, unfused) | 12.215 ± 0.069 | 12.141–12.279 | 12.759 ± 0.197 |
| **N = B + 48 null NKI calls** | **10.690 ± 0.039** | 10.665–10.736 | 11.484 ± 0.056 |
| **F = E with C's flags** | **12.179 ± 0.038** | 12.144–12.218 | 12.721 ± 0.185 |

- **Measured:** the bare NKI call boundary costs N − B = **+0.169 ms/step** (2 × pooled std 0.133), i.e. **≈ 3.5 µs per call** (48 calls), **10% of E's +1.69 ms**. Kevin's condition ("(i) accounts for most of E's penalty") is **not met**: the trn2 fusion mechanism is not call-boundary cost.
- **Measured:** C's compiler flags do not change E (F − E = −0.036 ms, within 2 × pooled std 0.112).
- **Measured (with T9):** the remaining ~1.5 ms/step (≈ 31 µs/layer) is device work of the attention stage when it runs as its own kernel: per T9, +14–17 µs DMA, +13 µs vector, +6 µs collectives and 1.7× B's spill per layer, against −18 µs scalar. Inside the megakernel the same stage costs ~39 µs/layer less (C ≈ B).
- **Hypothesis (what fusion saves on trn2, to test next):** state the standalone kernel must rebuild on every call — weights and layer-invariant constants loaded into SBUF per call, the mask and permuted RoPE scratch rebuilt per call (C hoists them out of its layer loop), and its own all-reduce-gather — rather than the boundary itself. A standalone E with the mask/RoPE build hoisted to one call per step would isolate the hoisting share; that needs a kernel-file change (Q3, needs approval).
- Same-session drift: B 10.52 here vs 10.49 in T8 the previous evening; C 10.36 vs 10.29.

## 2026-11-18 23:40 → 00:10 ET — H2.2 (early-finish item): expert-GEMM precision emulated in the CPU golden (Measured)

`~/runs/run_py.sh p1_h22 tools/h22_accum.py --variants fp32 bf16in bf16acc --out ~/runs/p1/h22/h22.json` (CPU, 27 min). The layer-by-layer fp32 golden is re-run with only the expert MLPs' gate/up/down projections replaced (`Qwen3MoeMLP.forward` patched at runtime): **bf16in** = inputs and weights rounded to bf16, fp32 accumulation, outputs rounded to bf16 (like PSUM-fp32 hardware); **bf16acc** = the same with the contraction accumulated in bf16 across 128-wide partial products. Compared with the frozen golden: KL at all 7,264 corpus positions, residual rel-L2 at the 512 sampled positions (`golden/layers.pt`). → `results/phase1/h22.json`.

| Variant | mean KL(golden ‖ variant) | ÷ XLA's mean KL (2.26e-3) | max KL | top-1 = golden | residual rel-L2 L0 / L24 / L47 |
|---|---|---|---|---|---|
| fp32 (harness self-check) | 3.0e-6 | 0.001 | 0.005 | 100% | 0.0000 / 0.0011 / 0.0046 |
| bf16in | 6.85e-4 | **0.30** | 0.169 | 99.37% | 0.0009 / 0.0244 / 0.0343 |
| bf16acc (128-wide) | 9.86e-4 | **0.44** | 0.166 | 99.24% | 0.0019 / 0.0284 / 0.0355 |

- **Measured:** the self-check reproduces the golden (KL 3e-6; residual noise floor 0.005 at L47 from fp32 CPU reduction order, 10× below XLA's 0.051).
- **Measured:** bf16 rounding of the expert GEMMs alone (fp32 accumulation) produces ~30% of XLA's total logit divergence; bf16 accumulation over 128-wide chunks raises it to ~44% (+44% relative). Input/weight rounding matters more than accumulation precision on this model — consistent with m1 (bf16 accumulation over 8-wide chunks on device, KL ratio 1.04) being barely detectable.
- Caveat: residuals are on the 512 sampled positions (one high-error position makes L33 read 0.098 for both variants vs XLA's all-position 0.067), so compare variants with each other, not with the all-position XLA table.

## 2026-11-18 18:50 ET — H2.6 (early-finish item): error vs depth for XLA, B, E (Measured)

`python tools/error_budget.py --builds xla=$HOME/runs/p1/val/xla_base B=$HOME/runs/p1/val/libmoe E=$HOME/runs/p1/val/E_cap --corpus … --out results/phase1/error_budget/xla_B_E` (7,200 decode positions each; residual = decoder-layer output, rel-L2 vs fp32 golden).

| Layer | XLA | B | E | B / XLA | E / XLA |
|---|---|---|---|---|---|
| 0 | 0.0074 | 0.0052 | 0.0051 | 0.70 | 0.69 |
| 8 | 0.0203 | 0.0163 | 0.0166 | 0.81 | 0.82 |
| 16 | 0.0303 | 0.0261 | 0.0263 | 0.86 | 0.87 |
| 24 | 0.0441 | 0.0408 | 0.0418 | 0.93 | 0.95 |
| 32 | 0.0526 | 0.0479 | 0.0503 | 0.91 | 0.96 |
| 36 | 0.0654 | 0.0670 | 0.0697 | 1.03 | 1.07 |
| 40 | 0.0521 | 0.0483 | 0.0503 | 0.93 | 0.97 |
| 47 | 0.0510 | 0.0500 | 0.0508 | 0.98 | 1.00 |
| peak (layer) | 0.0681 (L34) | 0.0758 (L33) | 0.0794 (L33) | | |

- **Measured:** the NKI MoE builds start with ~30% less residual error than XLA (lib MoE's layer-0 MoE output: 0.0107 vs 0.0165), and the advantage decays with depth: B/XLA 0.70 → 0.98 and E/XLA 0.69 → 1.00 by L47. Per-layer error injected by later layers is similar across builds and dominates the early difference.
- **Measured:** all three builds peak at L33–34 and then **damp** by 25–36% to L47; final-layer residual errors are within 2% of each other (0.050–0.051), consistent with the near-1 KL ratios.
- **Measured:** E's residual error is 0–5% above B's at every depth (largest around L32–36), the only accuracy cost of the standalone attention kernel.
- **Hypothesis:** the post-L34 damping reflects late layers (L37 attention, L33 MoE are the largest single-block errors) writing contributions that partly cancel accumulated error in the residual direction; testable by projecting per-layer error onto the residual.

## 2026-11-18 17:32–17:50 ET — T9 profile of config E, seq 768 (Measured)

```bash
bash ~/runs/prof/run_prof.sh p1_prof_E_s768 E ~/qwen-30b-a3b/traced_E_trn2_229_s768 768     # 13.8 min; profile_decode.py, kind E
python tools/profile_table.py ~/runs/prof/tables/E_s768.json E_s768_b128=…/tkg_bucket0.neff:…/small_bucket/*/*/*.ntff \
  E_s768_b768=…/tkg_bucket3.neff:…/large_bucket/*/*/*.ntff
```
Same method as Phase 0 Table 5 (inspect `model` mode around TKG calls 6 and 764, least-skewed rank, µs per layer). B, C, D rows are the Phase 0 tables (Nov 11–12).

| µs/layer (bucket 128 / 768) | B lib MoE | C megakernel | D lib MoE + NxDI NKI attn | **E** | E − B | E − C |
|---|---|---|---|---|---|---|
| Wall | 193.5 / 203.6 | 192.5 / 192.8 | 219.2 / 230.4 | **231.3 / 231.3** | +37.8 / +27.7 | +38.8 / +38.5 |
| Sync engine | 60.2 / 60.1 | 58.8 / 58.5 | 59.3 / 59.1 | 62.0 / 62.1 | +1.8 / +2.0 | +3.2 / +3.6 |
| DMA active | 126.1 / 128.3 | 123.3 / 123.5 | 146.0 / 151.3 | 142.7 / 142.3 | **+16.6 / +14.0** | +19.4 / +18.8 |
| Tensor engine | 62.5 / 62.8 | 58.6 / 58.4 | 66.2 / 65.6 | 64.9 / 64.7 | +2.4 / +1.9 | +6.3 / +6.3 |
| Vector / Scalar / GpSimd | 45.5 / 55.3 / 61.3 | 49.2 / 29.8 / 56.4 | 52.8 / 36.2 / 60.6 | 58.4 / 37.0 / 57.7 | **+12.9** / −18.3 / −3.6 | +9.2 / +7.2 / +1.3 |
| CC ops | 19.2 / 20.0 | 28.8 / 28.8 | 20.7 / 20.7 | 25.1 / 25.0 | +5.9 / +5.0 | −3.7 / −3.8 |
| DMA packets / avg bytes | 42,261 / 790 | 43,748 / 783 | 41,125 / 802 | 44,169 / 753 | +5% / −5% | +1% / −4% |
| Spill reload / save (bytes per graph) | 212,864 / 122,888 | 19,328 / 36,872 | 470,912 / 233,480 | 360,320 / 135,176 | 1.7× / 1.1× | 18.6× / 3.7× |
| Graph time (ms) | 9.29 / 9.77 | 9.23 / 9.25 | 10.52 / 11.06 | **11.10 / 11.10** | +1.81 / +1.33 | +1.87 / +1.85 |

- **Measured:** E's slowdown is device time: graph +1.81 ms (b128) / +1.33 ms (b768) over B, matching the benchmark gap (+1.65 ms at 768, averaged over buckets). Host overhead is not the explanation.
- **Measured:** E is identical in the 128 and 768 buckets (wall 231.3 µs, same packet count): its kernel, like C, attends over the full `kv_mgr.past_key_values` cache (768 slots, masked) in every bucket — which is also why E's and C's benchmark latency barely grows with context. B's attention grows +10 µs/layer from bucket 128 to 768.
- **Measured:** relative to B, E adds DMA-active +14–17 µs, vector +13 µs and CC +5–6 µs per layer (the in-kernel all-reduce-gather replaces XLA's), removes 18 µs of scalar work (XLA softmax/RoPE), and spills 1.7× more; relative to C, the same attention stage costs +39 µs/layer when run standalone, and C spills 19× less.
- **Hypothesis (T8b tests it):** most of E's +38 µs/layer is the cost of entering and leaving an NKI kernel per layer — reloading weights/constants into SBUF and the mask/RoPE scratch rebuilt per call, and no overlap with the neighbouring XLA ops — rather than the attention math itself.

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
- See T9 below: E's extra time is on the device (graph +1.8 ms over B at bucket 128).
- Day-to-day: B at 768 is 0.18 ms slower than on Nov 12 (10.31 ± 0.01, pre-sharded weights; today shard-on-load) and C 0.14 ms slower; same-session differences are what count.
