#!/usr/bin/env bash
# Parallel pre-read of lazily restored EBS blocks: every file in 1 GiB chunks, 16 readers.
date -u +%FT%TZ
find "$@" -type f -size +1M -printf '%s %p\n' | while read -r s p; do
  for ((o=0; o<s; o+=1073741824)); do echo "$((o/16777216)) $p"; done
done | xargs -P 16 -L 1 sh -c 'dd if="$1" of=/dev/null bs=16M skip="$0" count=64 status=none'
date -u +%FT%TZ
echo PREREAD_DONE
