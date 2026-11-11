"""Engine time by block (attention / MoE / other) from neuron-profile per-instruction JSON.

Computes (µs per layer, n_layers=48):
    per block x engine: summed instruction duration (ns units; overlaps, so an upper bound on busy time)
    and summed evt_wait_time (stall)
    per block: HBM read MB/layer

Usage:
    profile_breakdown.py PROF.json [PROF.json ...]   (generated with --output-format json)
"""
import collections, json, re, sys

N_LAYERS = 48
BLOCKS = [("attn", re.compile(r"attention|attn|kv_cache|rotary|rope", re.I)),
          ("moe", re.compile(r"mlp|moe|expert|router|interleave_copy", re.I))]


def block(i):
    s = str(i.get("nki_source_location") or i.get("bir_debug_info_source_location") or "").split(":")[0]
    for name, rx in BLOCKS:
        if rx.search(s):
            return name
    return "other"


def engine(i):
    lab = str(i.get("label") or i.get("subgroup") or "")
    return "DMA" if "dma" in lab.lower() or "queue" in lab.lower() else lab.split()[0] if lab else "?"


def main():
    for p in sys.argv[1:]:
        d = json.load(open(p))
        dur = collections.defaultdict(float)
        stall = collections.defaultdict(float)
        hbm = collections.defaultdict(float)
        for i in d["instruction"]:
            b, e = block(i), engine(i)
            dur[b, e] += i.get("duration", 0) or 0
            stall[b, e] += i.get("evt_wait_time", 0) or 0
            hbm[b] += i.get("hbm_read_bytes", 0) or 0
        out = {f"{b}/{e}": {"dur_us": round(dur[b, e] / 1e3 / N_LAYERS, 2), "stall_us": round(stall[b, e] / 1e3 / N_LAYERS, 2)}
               for b, e in sorted(dur)}
        out.update({f"{b}/hbm_read_mb": round(v / 1e6 / N_LAYERS, 3) for b, v in hbm.items()})
        print(p, json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
