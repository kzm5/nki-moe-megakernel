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
- 2026-11-23 19:55 ET — Q4 T3 done: V bit-identical across 5 processes. T5 (H2.4 emulation) running, then Eh compiles, T4 at 2048, T2.
- 2026-11-23 20:30 ET — Q4 T5 done (emulated bf16 AR +43% KL vs fp32 sum; device C ≈ V, bit-identical 62/64 prompts). Eh 2048 compile running.
- 2026-11-24 00:25 ET — Q4 T4 (2048) done: XLA/B/C/V all pass v1.1 at --seq-len 2048 (positions < 768). Eh compiled at 2048/4096; T2 bench running (~18 h). Disk 22 GB free.
- 2026-11-24 06:50 ET — Q4: T1/T3/T4(2048)/T5 done; T2 2048 done (Eh − C +0.40 ms, Eh − B −0.20 ms); T2 4096 running. DAILY written.
- 2026-11-24 16:50 ET — Q4 done. T2: Eh − C ≈ +0.36–0.40 ms at every length (does not grow); Eh beats B at 2048/4096 (attention). C = V bit-identical at 2048. Q5 T6 final session started (~30 h).
- 2026-11-24 17:10 ET — Q4b added: T6 stopped (no data lost); T9 long-context corpus started. **Waiting on Kevin:** deletion approval for T10 (`results/phase1/deletion_candidates_q4b.md`, Tier 1 = 22.8 GB).
- 2026-11-24 18:10 ET — Q4b T9 done: supplementary long-context validation at 2048, XLA/B/C/V all pass v1.1 rows. T11 package written (proof run next). T10 waits for disk approval (17 GB free).
- 2026-11-26 11:50 ET — deletions done (52.3 GB, approved); Q4b T10 (8192) running; T9 done; T11 written, validate proof pending. Session was idle Nov 24 18:15 → Nov 26 11:38 waiting on the deletion answer.
- 2026-11-26 21:50 ET — Q4b T10 done: at 8192 V 10.313 vs B 11.398 ms (−9.5%; lead keeps growing); C hangs at runtime at 8192 (device recovered by driver reload). Next: T11 validate proof, T4 at 4096, then Q5 T6.
- 2026-11-26 22:15 ET — Q4b T11 done (validate proof bit-identical 64/64). T4 at 4096 running (~3.5 h), then Q5 T6 (~30 h).
