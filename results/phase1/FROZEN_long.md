# Supplementary long-context corpus — frozen 2026-11-24 23:05 UTC

Separate from the v1 validator (`FROZEN.md`, spec v1.1, `corpus/`), which is unchanged. Hashes in `frozen_long.json`.

| Item | sha256 |
|---|---|
| `corpus_long/corpus.json` (16 prompts q00–q15, plen 1,400–1,895, 64 new tokens; 4 × 1,920-token ppl passages) | `a15eee9f71bfa0f8afa68bc202e6609cb060ac5bf2c3d5bef71369d7acb2cf9d` |
| golden manifest (`~/runs/p1/golden_long2048`, 21 files, fp32 CPU golden via frozen `golden.py`) | `700af57dd0ddb123cef94f44055e52e1cca33324fb0ad599358c006e68fd707c` |
| refs manifest (`~/runs/p1/refs_long2048`, 16 files, XLA 2048 free-running via frozen `device.py --free all`) | `79469a020e5f31b6d46eb1e6089bcc6cee2f9a586348055fd039b3dcf3b68ce2` |
| scoring wrapper `tools/validate_long.py` (imports the frozen `validate.py` unchanged; repoints CORPUS/GOLDEN/REFS) | `bf1da47ec8ac7d9f17593387d3983ec1378aa43d427aea722d1f0f5c2fed4562` |
| corpus builder `tools/corpus_long.py` (deterministic: rebuild gives the same bytes; seed 1124) | `5f218364f35136ae3595be00fd29bede657eddd4ae1eea435ef90f02dfdb0e48` |

Frozen validator files and spec: as in `frozen.json` (checked by the wrapper: `frozen_long_ok`).
Sources: wikitext-2-raw-v1 test (CC BY-SA 3.0), articles not used by the v1 corpus; CPython 3.12 standard library (PSF-2.0).
