# Agent status (fallback for GitHub issue "Agent status": issues are disabled on kzm5/nki-moe-megakernel)

- 2026-11-17 14:15 ET — validator frozen (T2/T3/T4 done); Gate-1 device chain running. Commit 86f54c0.
- 2026-11-17 21:30 ET — Q1 done: Gate 1 met (m1/m2/m4 caught, m3 passes); C passes, B fails worst_pos on 1 position (see FINDINGS); report in FINDINGS/DAILY. Commit aaddfe9.
- 2026-11-18 08:00 ET — daily note written; T6/T7 done (E built, passes all rows but worst_pos like B); T8 768 done (B 10.49, C 10.29, E 12.14 ms), 2048/4096 running.
- 2026-11-18 23:35 ET — Q2 done (T6–T11, T8b); report `results/phase1/SUMMARY_Q1Q2.md` written; device idle. Commit: see `git log` (pushed after this line).

- 2026-11-19 12:55 ET — Q3: T1 done; T2 done (fp32 router already in B/C/XLA; T2.2 skipped by rule); T3.1 done; T3.2 Eh passes v1.1 and recovers 32 of E's 38.6 µs/layer in the profile; bench running.
- 2026-11-19 22:45 ET — Q3 T3 done (Eh recovers 82% of E − C; 10.707 vs C 10.373 ms); mid summary written; T4 A1 agent run started (6 h).
- 2026-11-22 17:55 ET — Q3 resumed after the credit outage (no work Nov 21–22). C2 partial recorded (2 iters, both pass v1.1, deterministic, best 10.035 ms); C2r (restart) running 22:36–04:36 UTC; then the final B/C/V bench. Free disk 56 GB; deletion list awaiting Kevin.
- 2026-11-23 04:20 ET — Q3 T4: C2r done (best eligible 10.082 ms; iter 2 non-deterministic). Final bench V = C1 iter 2 (cross-process bit-identical): 768 n=5 V 10.050 ± 0.015, C 10.224 ± 0.007, B 10.500 ± 0.013 ms → V − B −4.3% (< Gate 4's 5%). 2048/4096 running. Free disk 20 GB.
- 2026-11-23 16:10 ET — Q3 done. V (C1 iter 2) 10.050 ms at 768: −1.7% vs C, −4.28% vs B (Gate 4 5% not met at 768; 7.61 / 8.57% at 2048 / 4096, unvalidatable builds). SUMMARY_Q3.md written; device idle; free disk 71 GB.
- 2026-11-23 17:55 ET — Q4Q5 started; device idle; T1 (merge V with asserts) next.
- 2026-11-23 18:45 ET — Q4 T1 done: V + asserts merged (a62b8af); outputs bit-identical to 35309c3's build 64/64. T3 captures running.
