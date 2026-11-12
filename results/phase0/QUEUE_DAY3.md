# Day-3 queue (verbatim from the user, 2026-11-11 23:2x ET)

Your Task 5 question: yes, bisect — but run a teacher-forced census of every position first, so the bisect
targets come from that census, not only from first divergences. This queue replaces the old one.

DEADLINES (all ET; the instance clock is UTC, so 20:00 ET = 01:00 UTC Fri). Print `TZ=America/New_York date` at
the start of every task.
- No new compile after 16:00 ET. No device job after 18:00 ET.
- SUMMARY_DAY3.md, RESUME.md, refreshed backup, final push and a clean tree by 19:30 ET. Hard stop 20:00 ET; I image the instance after that.
- If a sub-step is blocked for more than 45 min, record it (ledger + FINDINGS, labeled), take the stated fallback, and move on.
Rules as before: CLAUDE.md, one device job at a time, check free disk before every compile/load, log every deletion,
never delete ~/runs/golden_v2, ~/runs/val, ledger or results/, label each claim Measured or Hypothesis, push after each task.

T0 — CPU triage on existing data (~45 min; run it while T1's first model loads)
For every first divergence (gross and benign, both builds), using ~/runs/golden_v2 and the day-1 captures:
 a. Shift test: KL(build@p ‖ golden@p−1) and KL(build@p ‖ golden@p+1) vs KL(build@p ‖ golden@p). Is the NKI
    output a prediction for a neighboring position (lag/lead) rather than noise?
 b. Router margin: the golden 8th–9th router gap at each layer, min over layers, at gross vs benign vs random
    positions (H2.1 preview).
 c. Absolute position (prompt length + p) and TKG bucket for every gross position; flag any near a bucket edge.
Output: tools/triage_gross.py + ~/runs/triage/triage.json, one FINDINGS section.

T1 — Teacher-forced census of every position (top priority; budget ~6 h; aim to finish by ~06:30 ET)
 1. Write tools/capture_tf.py: prefill on the prompt, then a manual decode loop that feeds XLA's reference token
    at every step (ignoring the on-device sampled token) and saves the returned logits. Same flags as the greedy builds.
 2. Self-check: XLA build teacher-forced must reproduce its free-running logits bit-for-bit on all 20 prompts.
    Don't trust the harness until this passes.
 3. In the same XLA process, add 4 long prompts (prompt 500–600 tokens, 64 new tokens, total ≤ 768) and capture
    XLA's free-running output for them; then run golden_fp32.py on them (CPU).
 4. Library MoE and megakernel builds, teacher-forced over all 20 + 4 prompts (2,816 positions each). In the same
    process, run the 20 short prompts a second time (in-process determinism); compare positions before divergence
    with the day-1 free-running captures (cross-process determinism). This also answers the open question of whether the megakernel is deterministic.
 5. Still in each NKI process, single-step variant at every gross position: prefill on tokens[0..n−2], then exactly
    ONE decode step on token n−1. Same gross error as in the teacher-forced loop or not? (Does it depend on
    KV state written by earlier decode steps, or only on this step?)
 Census (tools/census.py): per position, KL(golden‖build), KL(golden‖XLA), rel-L2, top-1. Gross = KL(golden‖build) ≥ 0.2
 nats AND KL(golden‖XLA) < 0.01; report the KL histogram so the threshold is visible. Report the gross rate per build,
 how many gross positions the two builds share, short vs long prompts, ρ over all positions with and without gross
 positions, max per-position KL, and the determinism results.
 Fallback if the manual loop can't be made to work in 2 h: get every position from the single-step variant
 (prefill on the prefix + one decode step per position, ~2,800 calls per build); state that in the summary.

T2 — Per-layer bisect against golden (this is Task 5; budget ~8 h; must end by 16:00 ET for compiles)
 Targets: p01@15, p06@4, p07@45, plus up to 5 of the largest gross positions from the T1 census (prefer ones shared by both builds).
 Library MoE first: it is XLA plus one NKI kernel, so it isolates the defect most cleanly.
 1. Extend golden_fp32.py to dump, at the target positions only, per layer: attention-block output, MoE-block output,
    residual stream after the layer, full router logits (128). Cheap CPU run.
 2. Device side: build a library-MoE variant with NxDI tensor capture (main.py tensor_capture_config) on the same
    per-layer tensors in decode. First confirm the gross error still reproduces in this build at the same positions; if it
    disappears, that is a finding (capture changed scheduling) and go to step 4.
 3. Per layer: rel-L2 vs golden for library MoE and for XLA (capture XLA the same way if it fits in time), and their ratio. Report the first
    layer whose ratio jumps more than 10x, and within that layer whether attention output or MoE output departs first. Check router top-8 at that layer against golden.
 4. Fallback (time-box 3 h): run the library MoE TKG kernel standalone on device, feeding it golden's bf16 layer
    input at the failing position, and compare it with HF's MoE block on the same input across all 48 layers.
 If both routes fail, write exactly where each one got blocked and what the next block should try.
 Megakernel bisect only if library MoE localizes early and time remains.

T3 — Optional, only if T2 is finished and ≥ 2 h remain before 18:00 ET: profile config D (one capture) to locate its
 +36 µs/layer.

T4 — Wrap-up (start by 18:00 ET at the latest)
 - results/phase0/SUMMARY_DAY3.md: key tables (census per build, determinism, single-step variant, bisect per-layer
   table), top 3 findings, what failed, recommended next 3 experiments.
 - results/phase0/RESUME.md, written for a fresh instance from this image: what's on disk and its size (builds,
   NEFFs, shards, golden, captures, caches), which items are essential, venvs and versions, run_job.sh usage, how to
   rerun the census and the bisect, known gotchas (do_sample "", seq_len % 256, swap, RAM peaks, tools 2.30).
 - Refresh ~/runs_backup.tar.zst to include today's data. Update the ledger and FINDINGS. Push, confirm HEAD matches the remote,
   working tree clean, no device process running, tmux idle.
 - Final message: result line, commit hash, free disk, backup size, and anything I must do before imaging.

## Addendum (verbatim)

Addendum to this queue: this session's context is long and will likely auto-compact during the run. To stay safe:
1. Keep results/phase0/QUEUE_STATE.md as the source of truth: for each task, its status, the exact commands and
   paths in use, the build dirs that still have weights, and the next step. Update it and push it after every sub-step.
2. After any compaction, or whenever you're unsure, re-read CLAUDE.md, QUEUE_STATE.md, the queue prompt and the
   latest FINDINGS sections before acting. Don't work from memory of earlier tasks.
3. Keep a copy of this queue prompt as results/phase0/QUEUE_DAY3.md so it survives compaction word for word.
4. The deadlines are ET; the instance clock is UTC. Re-check `TZ=America/New_York date` before every compile and
   every device job.
