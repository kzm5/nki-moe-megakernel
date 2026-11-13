# Day-3b queue (verbatim from the user, 2026-11-12 ~19:40 ET)

Standing rule, effective now: docs/RESEARCH_PLAN.md is a mirror of Kevin's master plan doc, committed by Kevin's
planning assistant. Never edit, regenerate or delete it. Before every push, run
`git pull --rebase mine phase0/trn2-setup` so pushes from both sides stay fast-forward. Put any discrepancy between
it and FINDINGS in FINDINGS, not in that file. Add this rule to CLAUDE.md.

Extra queue (optional, unattended). Kevin is imaging the instance and downloading ~/runs_backup.tar.zst — do NOT
modify, move or regenerate that file.
Deadlines (ET; instance clock is UTC): no new compile after 02:30, no device job after 04:30, everything pushed and
idle by 05:00. The Capacity Block ends 06:30.

Task R — NKI router decisions vs golden (the first follow-up in SUMMARY_DAY3)
1. Build the capture variant that exposes the library MoE kernel's own top-8 expert ids and router logits during
   decode, at every decode bucket if possible (day 3 only got the smallest); record which buckets worked.
2. Teacher-forced over the same 2,816 positions: per layer and position, top-8 set agreement with golden for library
   MoE and for XLA; flip rate per layer; golden's 8th–9th router margin at every flip.
3. Report: the NKI-minus-XLA flip rate in pp (overall and per layer) against the spec's 0.5 pp router gate; the share
   of per-layer MoE error carried by flipped positions; whether the margin predicts flips. Also state where the
   worst-KL position (0.56) sits and what XLA's KL is there.
Rules as before (CLAUDE.md, QUEUE_STATE.md, one device job, disk checks, ledger, Measured/Hypothesis labels, push
after each step). Don't delete anything RESUME.md lists as essential. If the capture build fails twice, stop,
document, push.
Wrap-up: add a "Day 3b" section to SUMMARY_DAY3.md and update RESUME.md and QUEUE_STATE.md. Push, confirm the tree is
clean and the device is idle. Final message: result line, commit, free disk.
