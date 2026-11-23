# Queue TRN3 — one day on trn3.48xlarge (Nov 25–26)

Written by Kevin's planning assistant. Read-only for the agent. Run in its own Claude Code session on the trn3
instance (us-east-2), launched from a copy of the trn2 AMI. Work on branch **`trn3/day`** (create it from
`phase0/trn2-setup` at the V merge commit); push only that branch. Results go in `results/trn3/`, with state in
`results/trn3/QUEUE_STATE.md`.

## Goal

One controlled comparison: the same commit, SDK and validator on trn3 as on trn2. Gomes's Spring trn3 numbers used
SDK 2.27 on trn3pd98.3xlarge and stay prior work, not a baseline.

## Hard limits

- Instance time is about 24 h from launch. **Stop all device work 2 h before Kevin's stated end time** (ask him if it
  isn't in the launch message), then wrap up. Push after every step.
- Use **one Trainium3 chip only**, TP=4 at LNC=2 (4 logical cores), matching trn2. Restrict the runtime to one chip's
  cores (`NEURON_RT_VISIBLE_CORES`), and confirm with `neuron-ls` and a trivial NEFF that only those cores are used.
- One device job at a time. Use the frozen validator v1.1 unmodified. No kernel edits.

## Steps (in priority order; if time runs short, drop from the bottom)

1. **Bring-up, time-boxed to 2 h.** Follow `scripts/trn3_bringup.md`.
   - Record the instance type, `neuron-ls`, and the driver/runtime/collectives/tools versions.
   - If the AMI's driver doesn't support trn3, install the oldest trn3-capable Neuron driver/runtime. Record the
     exact versions (they will differ from trn2's 2.32 runtime; that's a stated caveat). Keep the venv on SDK 2.29.
   - **LNC=2 test first:** compile and run a minimal TP=4, LNC=2 NxDI model for `--platform-target trn3`. If LNC=2
     is rejected, record it, write the summary and stop: the kernels' sharding is LNC=2-specific, so no comparison
     is valid.
   - Pre-read the weights. This host has ample RAM, so shard on load.
2. **Compile at 768:** A (XLA), B, C, V (`MK_AGENT_VARIANT=1`). Then Eh at 768 if it compiles. Log any compile
   failure with its first error and skip that config.
3. **Benchmark at 768:** interleaved A → B → C → V (→ Eh), n=3 each, p50/p99.
4. **Validator census at 768:** XLA on trn3 as the baseline (ρ and the ratios are relative to same-hardware XLA),
   then B, C, V against the existing fp32 golden and corpus (`~/runs/p1/{golden,refs}`; both are hardware-independent).
5. **Profile:** one per-layer decode profile each for B and C at 768, in the trn2 Table-5 format. Note any engine
   names or NTFF differences.
6. **2048, if time allows:** compile B, C and V, then benchmark n=3.

## Wrap-up

Write `results/trn3/SUMMARY_TRN3.md` with:
- the versions table (trn3 vs trn2)
- the latency table next to the trn2 numbers from `results/phase1/SUMMARY_Q3.md`
- fusion's net gain (C vs B) and the collectives gain (V vs C) on each chip
- validator verdicts
- the profile comparison
- every caveat (instance type, driver/runtime versions, one-chip restriction)

Push `trn3/day`, leave the device idle, and end with a final message: commit, and "done, instance can be
terminated".
