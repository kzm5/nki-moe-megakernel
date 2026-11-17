# Frozen validator

Frozen 2026-11-17T19:06:14Z. Every validator report and ledger entry carries these hashes; `validate.py` sets `frozen_ok` only when they match `frozen.json`.

| Item | SHA-256 |
|---|---|
| `tools/validator/validate.py` | `e32cf29d3369cf494bd65173f9332383f2bbe83beeece00aa72c919a1495ed2a` |
| `tools/validator/device.py` | `e26dbb468a96b7bf992343a6bd043a8cbacf57907c1f29673b7a93dbc3e3379c` |
| `tools/validator/golden.py` | `5e2d0aecdcc8b2295315973183ebec8ccf0684b340adc7425b5fed4d40c72ae5` |
| `tools/validator/mutants.py` | `f4481b6f6a7f66c48b0b0559a60a577476d556338477b46fbf3f4fe09bc0c6f0` |
| `tools/validator/corpus.py` | `ec6634b67798c030d9e4d116c7b9bfd3f07c7bff3c6fe564a14fbaf9aeeb0773` |
| `results/phase1/spec.json` | `7142014972bad7d4df51dcfb4bbfb49d4e0ee2812edbbe22cd508e816fc7b8eb` |
| `results/phase1/corpus/corpus.json` | `8d21c5280ce29030d88f7d04fb75e2008304146d602fd16f48377b141cbd8d73` |
| golden manifest (81 files in `~/runs/p1/golden`) | `1cb69418cbe33170f8461b37d830773fde20ad0bce702bbe21962175134b408a` |
| refs manifest (64 files in `~/runs/p1/refs`) | `f3d60037a0ad8d69025df8162df14321a0fb143de670e0d5bd82f80f6c0ea30f` |

Read-only paths (chmod a-w; PreToolUse hook `tools/validator/guard_hook.py` denies edits):

- `/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2/tools/validator`
- `/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2/results/phase1/spec.json`
- `/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2/results/phase1/corpus/corpus.json`
- `/home/ubuntu/runs/p1/golden`
- `/home/ubuntu/runs/p1/refs`

Per-file hashes: `results/phase1/frozen.json`.
