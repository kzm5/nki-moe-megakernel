# Trainium Research Plan — Execution & Agent Automation

Nov 10, 2026 · @Kevin

## Fix these before anything else

Three claims in the proposal don't match the Spring 2026 reports, and the first two change what this semester is about. All numbers below come from Kevin Gomes's report (Qwen3-30B-A3B, bs=1, 640 output tokens, trn3, TP=4, LNC=2, SDK 2.27).

| Claim in proposal | What the data says | Fix |
| --- | --- | --- |
| "1.76x decode throughput over the **library** baseline" | 1.76x is token-gen throughput over plain **XLA** (69.7 to 122.7 tok/s). End-to-end is 1.68x (65.9 to 110.7). Over the library baseline (XLA + SDK NKI MoE kernel), the custom fusion is **1.15x** TKG (106.3 to 122.7) and 1.14x E2E. | Report 1.15x as the fusion result and 1.52x as the SDK MoE drop-in. Gomes's "remaining 0.24x" adds ratios; speedups multiply. |
| The 1.76x path is a validated result | Gomes's report says R5 was met (logits vs a **stored** XLA baseline at rtol = atol = 1e-2). The competition report says the 48-layer megakernel failed contest tolerance and stock NxDI was the only verified-correct path. Both can be true; they used different gates. | The open question is whether the megakernel is correct, and that depends on which gate. Phase 1 settles the gate before any new speedup claim. |
| Target hardware is Trainium2 | Every megakernel number is on trn3 (trn3pd98.3xlarge). Gomes's report treats MXFP4 microscaling matmul and the LNC=2 SDK MoE kernels as trn3-specific. | Decided: trn2 (next section). All baselines are re-measured on trn2; the Spring trn3 numbers are cited as prior work, not compared head to head. |

One correction from reading the code: the repo's trn3 validation path compiles the XLA baseline fresh each run, so it does not rely on a stored reference as Gomes's report implies. The problems with it are different, and are listed in the next section.

## Hardware decision: trn2

Move forward on trn2. The research questions (numerics, fusion granularity, agent effectiveness) need no trn3-only feature. trn2 wins on what limits a solo researcher running agents: capacity and parallel device slots.

| Factor | trn2 | trn3 |
| --- | --- | --- |
| Capacity | Broadly available | Capacity limits blocked compile/debug cycles in the competition's final round |
| Parallel device slots for agents | Planned instance is trn2.3xlarge (one chip): one device experiment at a time, so throughput comes from compiling candidates on host CPUs while the device runs | The Spring box (trn3pd98.3xlarge) was one chip: one experiment at a time |
| Deployment relevance | Project Rainier runs on Trainium2 | Newer, smaller footprint |
| State of the code | Gomes's latest work (Qwen3.6 megakernel) was built on trn2 | Where the Qwen3 1.76x was measured |
| Quantization (Phase 5) | FP8 only | Native MXFP4 matmul |

**What trn2 costs:** the Qwen3 megakernel has no published trn2 numbers, its trn2 accuracy has never been checked (below), and Phase 5 loses MXFP4. Phase 0 produces the first validated trn2 baseline, which is real work but also a clean starting point. If Zhiru's credits only cover trn3, switch; the plan otherwise holds.

**Repo findings ([nki-moe-megakernel](https://github.com/KevGomes1403/nki-moe-megakernel), latest commit Aug 17 2026) — fix in Phase 0:**

1. `main.py` sets `NEURON_PLATFORM_TARGET_OVERRIDE="trn3"` unconditionally at the top of `main()`, silently overriding `--platform-target trn2`. Fix this first.
2. On trn2, `--mode validate` exits with "not supported" and `evaluate_single` hardcodes `accuracy = 1`. **No trn2 result from this repo has ever been accuracy-checked**, including the README's trn2 support claim for Qwen3.
3. The trn3 check is NxDI `logit_validation` with a tolerance map of atol 1e-5, rtol 0.03–0.05 on logits, not the 1e-2 stated in the report. By default it runs on **one prompt** ("I believe the meaning of life is"), which is far too thin to call a kernel validated.
4. SDK mismatch: the READMEs say SDK 2.27, but the `nkilib` submodule pins Gomes's fork at branch `2.30_release` (a `2.31_release` also exists). The Qwen3 megakernel imports from `nkilib` (`SbufManager`, `BufferManager`, `transformer_tkg`), so pin the SDK that matches the fork branch you check out.
5. The submodule URL is SSH (`git@github.com:...`). The fork is public, so switch it to HTTPS with `git submodule set-url` on the server.
6. Already present and worth reusing: NxDI tensor capture is wired into `main.py` (`tensor_capture_config`, dumps to `NKI_DEBUG_DUMP_DIR`). Phase 1 capture hooks build on this.
7. Skills: the repo has both `nki-kernel-optimizer` (trn2) and `nki-kernel-optimizer-trn3`, and the trn3 one declares itself the repo default. Flip that, or agents will load trn3 hardware facts on a trn2 box.
8. Out of scope but present: a Qwen3.6-35B-A3B speculative-decoding megakernel (DeltaNet hybrid, trn2).

## Phase 0 results so far (Nov 10)

On trn2 the SDK's library MoE kernel delivers nearly all of the decode speedup: 1.45x over XLA. The megakernel adds 1.02x on top at 768 tokens and 1.06x at 2048. Neither NKI build passes logit validation against XLA, but the two agree with each other. That makes the fp32 golden reference the next deciding experiment. All numbers below come from the fair rebuild on Nov 11: on-device greedy sampling and logits returned for every config.

| Config (trn2.3xlarge, SDK 2.29, seq\_len 768, bs=1, TP=4, LNC=2) | Decode p50 | Decode p99 | vs XLA | Passes logit check vs XLA |
| --- | --- | --- | --- | --- |
| XLA | 14.91 ms | 15.28 ms | 1.00x | reference |
| XLA + SDK library MoE kernel | 10.33 ± 0.04 ms (5 runs) | 11.09 ms | 1.45x | 0 of 20 prompts |
| Megakernel (Gomes) | 10.11 ± 0.02 ms (5 runs) | 10.59 ms | 1.48x | 0 of 20 prompts |

**The megakernel's lead over library MoE is real but small, and it is not yet attributable to fusion.** The gap is 0.215 ms per step, about 10 times the run-to-run spread. The megakernel's decode time stays near 10.2 ms from 256 to 2048 tokens, while library MoE grows from 10.21 to 10.84 ms, so the lead widens with context: none at 256 tokens, 1.06x at 2048. But in the library-MoE config, attention still runs in XLA. Context-dependent growth is an attention cost, so the megakernel's long-context advantage may come from its NKI attention kernel rather than from fusion. The Spring trn3 1.15x has the same confound. Separating the two needs a fourth config: library MoE plus an unfused NKI attention kernel.

**Correctness: the XLA reference may be the outlier (Hypothesis).**

- Most divergence points are bf16 near-ties.
- Peak logit errors reach 3.5–7.7% of the max logit.
- At two positions, both NKI builds pick the same unlikely token even though they use different MoE kernels.

Only an fp32 golden reference can say which build is more accurate.

**Process findings for Section 7:**

- The unattended agent ran 9 h 24 min and wrote 35 ledger lines.
- It caught a sampling bug on its own: `--top-k 1` alone is not greedy, and 8.9% of tokens were not the argmax. It redid every build before validating anything.
- Profiling was blocked by a tool-version mismatch: the runtime writes NTFF v7, which aws-neuronx-tools 2.28 can't read.

**What it took to run the Spring code on trn2.** These belong in the report's reproducibility section.

1. The only working combination is SDK 2.29 with fork branch `2.29_release`. The legacy MoE kernels need an nkilib field that was removed in 2.30, and the Qwen3.6 code needs `DtypeMode`, which only exists from 2.31.
2. Two hardcoded trn3 targets had to be removed, one in `main.py` and one in `qwen_with_megakernel.py` (applied on import). No trn2 megakernel build had ever loaded before this fix.
3. `nki_kernels/__init__.py` imported the Qwen3.6 kernels eagerly, which coupled every model to the Qwen3.6 dependencies. It is now lazy.
4. Megakernel attention at LNC=2 requires seq\_len to be a multiple of 256, so 640 fails.
5. Host RAM peaks at 121 GiB of 124 during compile and load. The fix is a 64 GB swapfile plus pre-sharded checkpoints.
6. The submodule pin `72ab554` was never pushed upstream.

**Where everything lives:**

- branch `phase0/trn2-setup` on Kevin's fork
- `results/phase0/FINDINGS.md` (lab notebook) and `SUMMARY_2026-11-11.md`
- `results/phase0/ledger.jsonl`
- raw logs under `~/runs` on the instance

## Thesis and scope

**Thesis:** for MoE decode on Trainium, measure numerical divergence against an fp32 golden reference rather than against bf16 NxDI. Then attribute that divergence to specific mechanisms, and find which fusion and precision designs keep the speedup inside a defensible error bound. An agent loop does most of the iteration and is built so it cannot game the validator.

The competition framed correctness as reproducing NxDI's exact bf16 rounding. That was the right target for passing a contest gate and the wrong one for research: it can reject a kernel that is *more* accurate than the baseline. With the competition over, the semester can redefine correctness, and that redefinition is itself a contribution.

The five goals were written for a team. Solo, they become one critical path, one conditional branch, and one method:

| Proposal goal | Role now | Why |
| --- | --- | --- |
| 1. Validation infrastructure | Critical path, first | Nothing downstream is trustworthy without it |
| 2. MoE precision path | Core research | Known unresolved issue; first real test of the framework |
| 3. Fusion / megakernel | Core, gated | Bounded payoff: fusion is worth about 1.15x over the library baseline, so it must stay correct to be worth it |
| 4. Quantization | Conditional | Only after Goals 1 and 3 pass their gates; needs its own quality-based acceptance criterion |
| 5. Agent-assisted optimization | Method across all phases | Becomes a result only if instrumented and compared across setups (see the agent-effectiveness section) |

**Out of scope this semester:** the model-translation skill and new ports, the GPT-OSS megakernel, the batch-8 regression, and the Trainium-vs-NVIDIA comparison. **One model, one operating point:** Qwen3-30B-A3B, batch 1 decode, fixed TP/LNC, one pinned SDK version.

## Correctness spec (take to Zhiru first)

A candidate kernel is correct if its error against an fp32 golden reference is no worse than stock NxDI's own error against that golden, within a margin. Bit-matching NxDI is reported but not required. Get this signed off before Phase 2; every result after it depends on it.

**Two references**

- **Golden:** an fp32 CPU forward pass of Qwen3-30B-A3B, run layer by layer from captured inputs, so the whole 30B model never has to sit in RAM at once. Use fp64 for spot checks.
- **Baseline:** stock NxDI in bf16 on the same instance and SDK version, captured fresh each campaign, never from a stored file.

**Key metric: error ratio.** For each captured tensor:

```latex
\rho = \frac{\lVert x_{cand} - x_{gold} \rVert_2 \,/\, \lVert x_{gold} \rVert_2}{\lVert x_{NxDI} - x_{gold} \rVert_2 \,/\, \lVert x_{gold} \rVert_2}
```

ρ ≤ 1 means the candidate is at least as accurate as the baseline. ρ much greater than 1 at one boundary points to the op that diverged.

**What gets measured**

| Level | Metric | Gate (strawman, set with Zhiru) |
| --- | --- | --- |
| Op boundary, every layer | rel-L2, max-abs, cosine vs golden; ρ | median ρ ≤ 1.25 per boundary, no layer above 2 |
| Router, every layer | top-8 expert-set agreement with golden; flip rate | flip rate no worse than NxDI's + 0.5 pp |
| Logits, teacher-forced | per-position KL(golden ‖ cand), top-1 agreement | mean KL no worse than NxDI's x 1.25 |
| Model quality | perplexity on a fixed eval set | delta vs NxDI within 0.5% |
| Run-to-run | two identical runs | bit-identical, or nondeterminism documented |
| Free-running greedy output | first-divergence position vs NxDI | reported only, not a gate |

**Why teacher-forced, not free-running:** in free-running decode, one near-tie argmax flip makes every later token differ. That makes token-match rate a chaos measurement rather than a correctness one. Teacher forcing feeds both models the same reference tokens, so each position's error is measured independently.

**Quantized kernels get a separate gate.** They cannot match bf16, so for them the gate is perplexity delta plus one downstream task, agreed with Zhiru before Phase 5 starts.

**Eval corpus:** a fixed, hashed set of prompts at varied lengths and positions, with fixed seeds. The validator, spec, golden tensors and corpus are frozen and read-only to agents.

## Execution plan

Six phases run in order, each closed by a gate; the agent harness runs alongside all of them. Gate 1, the signed-off correctness spec, is the one everything else depends on.

&#91;embedded content: execution roadmap · 6 phases, 5 gates\]

Gate criteria are spelled out in the backlog below. Phase 5 starts only if Gates 1 and 3 have passed. If Gate 4 fails, fusion work stops and becomes a written negative result.

## Experiment backlog

Each phase lists its experiments in run order and ends with the exit criterion its gate checks. The hypotheses in Phase 2 are the research core; the rest is setup for them or follow-through.

### Phase 0 — Reproduce and re-baseline

1. Pin instance type, SDK version and compiler flags. Reproduce three configs: XLA, XLA + SDK NKI MoE, megakernel.
2. Run each config 10 times to set a noise floor. No speedup claim smaller than that floor counts.
3. Audit the inherited skills (nki-kernel-optimizer, neuron-profile) by running each on a known kernel and checking its output.
4. If the instance is Trn2: test whether the megakernel compiles at all, and note what has to change (SBUF budget, no MX matmul).

**Exit:** all three configs reproduce within 5% of the Spring numbers on the same hardware, or the gap is explained.

### Phase 1 — Validation framework

1. Build the fp32 golden, layer by layer.
2. Add capture hooks at op boundaries, reusing the competition's list: input RMSNorm, Q/K projection, Q/K RMSNorm, RoPE, QK scores, softmax, attention out, all-reduce, router logits, top-k ids, expert outputs, weighted sum, MoE all-reduce, residual.
3. Profile NxDI's own error vs golden across all 48 layers. This is the error budget everything else is judged against.
4. Mutation-test the validator: inject known faults (bf16 accumulation, a dropped scale, a reordered expert sum) and confirm each one is caught. A validator that has never caught an injected fault proves nothing.
5. Compare the stored XLA reference from Spring against a fresh NxDI run.

**Exit:** the spec is signed off, the NxDI error profile is captured, and every injected fault is detected.

### Phase 2 — Divergence characterization

- **H2.1 Router flips dominate MoE divergence.** Top-k routing is discontinuous: a tiny hidden-state error near a router decision boundary swaps an expert and changes the layer output by a large step. Measure flip rate per layer, and how much of the output error comes from flipped tokens. Test whether the router margin (gap between the 8th and 9th logit) predicts which tokens flip. This is MoE-specific and likely the strongest finding.
- **H2.2 Accumulation precision.** Compare fp32 vs bf16 PSUM accumulation in the expert GEMMs, measured against the golden.
- **H2.3 Reduction order.** Vary the order of the expert weighted sum and of the RMSNorm reduction.
- **H2.4 All-reduce.** Compare bf16 vs fp32 all-reduce, and different reduction orders.
- **H2.5 Softmax implementation.** Compare divide vs reciprocal-multiply, and exp variants.
- **H2.6 Error across depth.** Is per-layer error amplified or damped through the 48-layer residual stream? Plot error vs layer for NxDI and the megakernel.
- **H2.7 Attribution by swap.** Swap one subkernel at a time (attention only, MoE only, router only) into stock NxDI.

**Exit:** a table attributing the megakernel's divergence to named mechanisms, with each share measured.

### Phase 3 — Correct MoE path

1. Fix the MoE precision path until it passes the spec.
2. Measure the latency cost of each precision choice and plot latency against ρ as a Pareto front.

**Exit:** a spec-passing MoE kernel, with the latency cost of correctness quantified.

### Phase 4 — Fusion granularity sweep

1. Fuse N layers per call, N ∈ {1, 2, 4, 8, 16, 24, 48}, with the residual passing through HBM between calls. For each N, measure TKG latency, compile time, compiler peak memory and ρ.
2. Batch all-reduces across layers. The megakernel's collective time rose 13.45 to 27.77 µs per layer (+106%), so about 14 µs per layer is recoverable.

**Exit (go/no-go):** some N passes the spec and beats XLA + SDK NKI MoE by at least 5% TKG. If none does, stop fusion work and write up the negative result. A clean negative result with a mechanism behind it is publishable.

### Phase 5 — Quantization (conditional)

1. FP8 experts with scales packed contiguously with their weights. Gomes's untested hypothesis is that this removes the per-expert scale DMA descriptor overhead.
2. MXFP4 experts, on trn3 only.
3. Gate on the quantized quality criterion, not on matching NxDI.

## Agent automation on the server

Start from Gomes's loop (the nki-kernel-optimizer and neuron-profile skills) rather than building from scratch. It is missing three things: a validator agents cannot touch, a device queue, and a ledger. Build those first.

&#91;embedded content: agent harness · roles, guardrail, scripts\]

Correctness is decided by a script, never by an agent. An LLM grading its own kernel is the failure mode this design removes.

**Roles**

- **Kevin:** queues objectives and makes architecture-level calls. Gomes found the loop weak at those.
- **Orchestrator:** a headless Claude Code run (`claude -p`) that takes one objective from a queue file and dispatches subagents defined in `.claude/agents/`.
- **Kernel optimizer:** extends nki-kernel-optimizer. Each attempt runs in its own git worktree, and only you merge to main.
- **Divergence bisector (new):** given a failing validator report, narrows the failure to a layer and op using the Phase 1 captures.
- **Profile analyst:** the neuron-profile skill, unchanged at first.
- **Experiment runner:** executes sweeps and the A/B/C setup comparison, and writes to the ledger.

**Guardrails**

- The validator, spec, golden tensors and eval corpus are owned by a different Unix user and are read-only. A `PreToolUse` hook also denies any Edit, Write or Bash command that touches those paths.
- Every validator result records the SHA-256 of the validator and the spec. The ledger rejects entries whose hashes don't match the frozen versions.
- A `Stop` / `SubagentStop` hook blocks any objective from ending as a success unless the ledger has a passing entry for it.
- Each objective has a budget: maximum iterations, wall time and device-hours.

**Throughput**

- Compile and run are separate stages. neuronx-cc compiles on host CPUs in parallel; device runs go through a single `flock` queue, or through `NEURON_RT_VISIBLE_CORES` partitions when a kernel needs fewer cores.
- A shared compile cache, keyed by kernel hash, avoids recompiling identical candidates.
- Objectives are queued overnight, and the ledger is summarized each morning.

**Ledger entry (one line per attempt)**

```json
{"run_id": "r0412", "objective": "moe-fp32-psum", "setup": "C", "agent": "kernel-optimizer",
 "parent_run": "r0409", "git_sha": "…", "hypothesis": "fp32 PSUM in down-proj cuts rho below 1.25",
 "compile": {"ok": true, "seconds": 412}, "validator": {"pass": false, "rho_median": 1.41, "rho_max_layer": 31,
 "kl_ratio": 1.08, "router_flip_pp": 0.3, "spec_sha": "…"}, "perf": {"tkg_p50_ms": 8.31, "n_runs": 10},
 "wall_s": 1630, "device_s": 240, "tokens": 182000, "cost_usd": 0.0, "human": false, "outcome": "fail"}
```

The values above are illustrative only. The ledger doubles as the dataset for the agent-effectiveness study.

**Build order:** device queue, ledger and a validator stub first; then hooks; then wire in Gomes's two skills; then the bisector, once Phase 1 captures exist.

## Agent effectiveness as a result (Goal 5)

Goal 5 becomes research only with a controlled comparison: the same fixed tasks, run under harness setups that differ in what feedback the agent gets, scored by the frozen validator. The anecdote from Spring ("agents sped things up") is not a result.

**Harness setups compared**

| Setup | Agent gets | Tests whether |
| --- | --- | --- |
| A | compile errors + pass/fail from the validator | the base loop works |
| B | A + profile-analysis skill | hardware feedback helps performance |
| C | B + per-boundary ρ report + divergence-bisector | localized numerical feedback helps correctness |

**Task suite (fixed, versioned)**

1. Fix an injected precision bug in the MoE kernel.
2. Optimize one subkernel to a target latency without failing the spec.
3. Localize the divergence in a deliberately perturbed megakernel.
4. Convert a single-layer kernel to N-layer fusion.

Run each task 3 times per setup, because agent runs are nondeterministic.

**Metrics, all read from the run ledger:** spec-pass rate, iterations to first pass, wall time, device-hours, tokens and dollar cost, validator-gaming attempts blocked by hooks, and human interventions. If time allows, do tasks 1 and 3 by hand once each, to get a human baseline.

**Also test Gomes's claim** that the loop handles local optimization but not architecture-level decisions. Task 4 is the probe: it requires choosing a structure, not just tuning one.

## Next-session decision tree

Three overnight results decide the next step:

- Does the megakernel still match library MoE under identical greedy settings?
- Do both NKI paths pass the correctness check against XLA?
- What does the per-layer profile show?

| If the overnight run shows | It means | Next step |
| --- | --- | --- |
| Megakernel within noise of library MoE | Fusion adds nothing on trn2 | Phase 4 becomes: why is fusion's benefit hardware-dependent? Compare the trn2 profile against Gomes's trn3 Table 5. |
| Megakernel at least 5% faster than library MoE | The Nov 10 tie was a sampling artifact | Phase 4 continues as planned with the fusion-depth sweep. |
| Library MoE fails the check at rtol 1e-2 | AWS's own kernel, the source of all the speedup, diverges from XLA | This becomes the first Phase 2/3 target. Bisect by layer with tensor captures. |
| Both NKI paths pass | No gross bugs | Build the Phase 1 fp32 golden reference and characterize error rather than hunt bugs. |
| Megakernel fails, library passes | Fusion is no faster and less accurate on trn2 | Write it up as a negative result and shrink Phase 4. |
| Profile shows SBUF spill or lost double-buffering | trn2's smaller SBUF breaks the fused design | Test smaller fused regions on trn2 (Phase 4 item 1). |
| Profile shows collectives dominate | All-reduce overhead cancels the fusion savings | Batch all-reduces across layers (Phase 4 item 2). |
| Queue incomplete or failed | — | Finish the queue Wednesday. Image the instance Wednesday night regardless. |

## Report outline

The final report argues two things. First, correctness has to be defined against an fp32 golden reference, not against bit-matching the baseline. Second, where the speedup comes from depends on the hardware. Each section below is tied to the evidence that backs it, so gaps show early.

| Section | Claim it supports | Evidence | Status |
| --- | --- | --- | --- |
| 1. Problem and contributions | — | — | Draft after Phase 2 |
| 2. Background: Trn2 vs Trn3, NxDI, NKI, MoE decode | — | Docs, Spring reports | Can write now |
| 3. Reproducibility audit | Prior results were not portable or validated on trn2 | Phase 0 fixes list, ledger | Mostly done |
| 4. Correctness methodology | fp32 golden plus error ratio beats bit-matching | Spec, validator mutation tests | Phase 1 |
| 5. Where the speedup comes from | Library MoE vs fusion, trn2 vs trn3 | Three-way tables, per-layer profiles | Overnight run |
| 6. Divergence characterization | Named mechanisms (router flips, accumulation, reduction order) | Phase 2 ablations | Phase 2 |
| 7. Agent-assisted workflow | What the harness enables, measured | Ledger, overnight transcript, A/B/C setups | Started Nov 10 |
| 8. Limitations and future work | — | — | End |

The Nov 10–11 overnight run is the first data point for Section 7. Keep its Claude Code transcript along with the ledger.

## Risks and open questions

The biggest risk is doing this solo: if the harness grows into its own project, the research never starts. Build the minimum harness in Phase 0, then extend it only when a phase needs it.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Hardware differs from Spring (Trn2 vs trn3), or capacity is short | Baselines don't carry over; device time is the bottleneck | Capacity Blocks end on a fixed date and terminate the instance: code in git, weights and captures on a volume that survives termination plus S3 snapshots, ledger synced off-instance |
| SDK has drifted from 2.27 | Reproduction fails for reasons unrelated to the kernels | Pin the SDK; reproduce on 2.27 first, upgrade as a separate controlled step |
| Megakernel code knowledge left with Gomes | Slow debugging of code you didn't write | Phase 0 audit; have an agent write a design-notes file from the code, then review it yourself |
| Golden fp32 run is too slow or memory-heavy on CPU | Phase 1 stalls | Layer-by-layer from captured inputs; a subset of layers and positions for the full sweep |
| Zhiru wants bit-exact NxDI matching | The fp32-golden framing is rejected | Report both; argue with H2 data that bit-matching can reject more accurate kernels |
| Agents game the validator | False passes | Frozen validator, hooks, hash checks (agent architecture section) |

**Open questions**

- [x] Instance: trn2.3xlarge in sa-east-1b. It has 1 chip, 4 logical cores at LNC=2, 12 vCPUs and 124 GiB RAM, and runs on a Capacity Block ending Nov 13 at 6:30 AM ET.
- [x] SDK: 2.29, in `~/venv-neuron-229`, with nkilib fork `2.29_release`.
- [x] Code: `KevGomes1403/nki-moe-megakernel`, forked to Kevin's GitHub. Work is on branch `phase0/trn2-setup`.
- [x] Claude Code runs unattended on the instance inside tmux.
- [x] Credits: not a binding constraint (Kevin, Nov 10).
- [ ] Does Zhiru accept the fp32-golden correctness definition?
- [ ] Will Gomes push submodule commit `72ab554`?
