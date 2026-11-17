# Phase 1 findings (lab notebook, Q1+Q2 onward)

Queue: [`docs/queues/Q1Q2.md`](../../docs/queues/Q1Q2.md). State: [`QUEUE_STATE.md`](QUEUE_STATE.md). Ledger: `~/runs/ledger.jsonl` → [`ledger.jsonl`](ledger.jsonl). Times ET unless marked UTC.

## 2026-11-17 11:05 — State on start (Measured)

- Commit `ffeed5c` (branch `phase0/trn2-setup`); work in worktree `.claude/worktrees/q1q2`, pushed with `git push mine HEAD:phase0/trn2-setup`.
- Disk: 160 GB free of 484 GB (`/`); RAM 121 GiB available of 124; swap 64 GB `/swapfile`, 0 used.
- `neuron-ls`: trn2.3xlarge `i-0f6abedd480e1a691`, 1 device, 4 logical cores, LNC=2, 96 GB HBM, idle. Runtime-lib 2.32.31, collectives 2.32.28, dkms 2.28.0, tools 2.30.10.0 (same as Phase 0).
- `gh auth status`: logged in as the repo owner (scopes repo, workflow).
- Builds present: XLA / lib MoE / megakernel `*_s768_greedy`, `traced_{xla,xlamoe}_*_s768_capture`, config D s768/s2048, sweep builds s256–4096, lib MoE router attempt-1 build `traced_xlamoe_*_s768_router`. None have pre-sharded `weights/`.
- **Missing from the restored volume:** all Day-3b output — `~/runs/router` is empty, and `traced_xlamoe_*_s768_router2` and `traced_xla_*_s768_router` are absent. The snapshot predates Day 3b (~19:40 ET Nov 12). `~/runs/ledger.jsonl` had 84 lines; the repo copy (88, a strict superset) was restored over it (logged in `~/runs/deletions.log`). The Day-3b results survive in git (FINDINGS, SUMMARY_DAY3); router reruns need the capture builds recompiled.
- `~/runs/run_{py,tf,job}.sh` and `ledger_event.py` now point to this worktree instead of `session-rules`.
