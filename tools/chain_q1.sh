#!/usr/bin/env bash
# Q1 device chain after the freeze, one job at a time. Each line: NAME KIND CDIR [device.py flags...]
WT=/home/ubuntu/nki-moe-megakernel/.claude/worktrees/q1q2
C=$WT/results/phase1/corpus/corpus.json
Q=/home/ubuntu/qwen-30b-a3b
V=/home/ubuntu/runs/p1/val
SHA=$(git -C $WT rev-parse --short HEAD)
run() {
  name=$1; kind=$2; cdir=$3; shift 3
  if [ -f $V/$name/device.json ]; then echo "skip $name"; return; fi
  df -h / | tail -1 | tee -a ~/runs/p1/chain_q1.log
  echo "$(date -u +%FT%TZ) start $name" >> ~/runs/p1/chain_q1.log
  ~/runs/run_py.sh p1_$name tools/validator/device.py $kind $cdir $V/$name --corpus $C --refs ~/runs/p1/refs --tf --ppl --repeat "$@"
  st=$(grep -o "EXIT_STATUS=[0-9]*" ~/runs/p1_$name.log | tail -1)
  python3 ~/runs/ledger_event.py p1_${name}_$(date -u +%Y%m%d) ~/runs/p1_$name.log status=$([ "$st" = EXIT_STATUS=0 ] && echo ok || echo failed) \
    kind=validator_device phase=1 "config={\"kind\":\"$kind\",\"build\":\"$cdir\",\"flags\":\"$*\"}" \
    "frozen=$(cat $WT/results/phase1/frozen.json | python3 -c 'import json,sys;d=json.load(sys.stdin);print(json.dumps({k:d[k] for k in ("spec","corpus","golden_manifest","refs_manifest")}))')" > /dev/null
  echo "$(date -u +%FT%TZ) end $name $st" >> ~/runs/p1/chain_q1.log
}
for job in "$@"; do
  case $job in
    xla_self) run xla_self xla $Q/traced_xla_trn2_229_s768_vcap --capture ;;
    m1|m2|m3|m4) run $job xla $Q/traced_xla_trn2_229_s768_vcap_$job --capture --mutant $job ;;
    libmoe) run libmoe libmoe $Q/traced_xlamoe_trn2_229_s768_vcap --capture ;;
    nki) run nki nki $Q/traced_nki_trn2_229_s768_greedy ;;
    libmoe_plain) run libmoe_plain libmoe $Q/traced_xlamoe_trn2_229_s768_greedy ;;
    *) echo "unknown $job" ;;
  esac
done
echo "$(date -u +%FT%TZ) CHAIN_DONE $*" >> ~/runs/p1/chain_q1.log
