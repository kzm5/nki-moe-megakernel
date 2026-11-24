# Deletion candidates for Q4b T10 (8192 builds) — awaiting Kevin's approval

Posted 2026-11-24 17:10 ET. Free disk 22 GB of 678. T10 needs ~4 builds at 8192 (≈3.5–4 GB each, Hypothesis from 4096's 2.7 GB) plus compile-cache growth (≈5–10 GB) and T9's outputs (≈3 GB). Nothing below is deleted until approved.

## Tier 1 — redundant (results kept in the repo)
| Path | Size | Why it is redundant |
|---|---|---|
| `~/runs/p1/val/agent_c1_2_rerun3`, `_rerun4`, `_rerun5` | 4.2 GB each (12.6) | Q4 T3 captures; comparison JSONs in `results/phase1/t4/c1_2_r2r{3,4,5}*.json` (all bit-identical to capture 1) |
| `~/runs/p1/val/V_t1_s768` | 4.2 GB | Q4 T1 rebuild outputs; report in `t4/reports/V_t1_s768.json`, comparison in `t4/V_t1_vs_c1_2.json` (bit-identical) |
| `~/qwen-30b-a3b/traced_V_t1_s768` | 1.7 GB | T1 rebuild of V (same outputs as `agent_c1_2`, which T6 uses) |
| `/tmp/nxd_model` | 4.3 GB | last compile workdir; rewritten by the next compile |
| **Tier 1 total** | **22.8 GB** | |

## Tier 2 — Q3 agent-study artefacts (records kept)
| Path | Size | Note |
|---|---|---|
| `~/runs/p1/val/agent_{a1_3,a2_4,base_1,c2_2,c2r_1}` | 4.2 GB each (21) | scoring dirs; reports now in `t4/reports/` |
| `~/qwen-30b-a3b/agent_{a1_3,a2_4,base_1,c2_2,c2r_1}` | 1.7 GB each (8.5) | non-V agent builds; diffs and branches pushed |
| **Tier 2 total** | **29.5 GB** | |

## Not proposed
C's pre-sharded `weights/` (59 GB; T6 loads it), compile cache (92 GB; T10/T6 reuse it), golden/refs, every build used by T6 (A, B, C, E, Eh, V at 768/2048/4096), `~/runs_backup.tar.zst` (13 GB).

Recommendation: Tier 1 is enough for T10 (22 + 22.8 ≈ 45 GB free).
