"""Per-layer decode profile table (Table 5 columns) from neuron-profile summary-json.

Computes (least-skewed rank = min total_time, divided by n_layers):
    wall, sync engine, DMA active, tensor engine, CC ops (all µs/layer),
    DMA packets/layer and average packet bytes (hardware + software + static DMA)

Usage:
    profile_table.py OUT.json LABEL=NEFF:NTFF_GLOB [LABEL=NEFF:NTFF_GLOB ...]
"""
import glob, json, subprocess, sys

N_LAYERS = 48
TIME_KEYS = {"wall_us": "total_time", "sync_us": "sync_engine_active_time", "dma_active_us": "dma_active_time",
             "tensor_us": "tensor_engine_active_time", "cc_us": "cc_op_active_time",
             "vector_us": "vector_engine_active_time", "scalar_us": "scalar_engine_active_time",
             "gpsimd_us": "gpsimd_engine_active_time"}
PKT = ("hardware_dynamic_dma", "software_dynamic_dma", "static_dma")


def summary(neff, ntff):
    out = subprocess.run(["neuron-profile", "view", "-n", neff, "-s", ntff, "--output-format", "summary-json"],
                         capture_output=True, text=True, check=True).stdout
    return list(json.loads(out).values())[0]


def row(neff, ntffs):
    allr = [summary(neff, f) for f in ntffs]
    # Ranks enter a profiled step staggered; early ranks idle inside collectives. Use the last starter.
    rs = [min(allr, key=lambda s: s["total_time"])]
    r = {k: sum(s[v] for s in rs) / len(rs) * 1e6 / N_LAYERS for k, v in TIME_KEYS.items()}
    r["all_rank_graph_ms"] = [round(s["total_time"] * 1e3, 2) for s in allr]
    pk = sum(sum(s[f"{p}_packet_count"] for p in PKT) for s in rs) / len(rs)
    sz = sum(sum(s[f"{p}_size"] for p in PKT) for s in rs) / len(rs)
    r.update(graph_ms=r["wall_us"] * N_LAYERS / 1e3, dma_packets_per_layer=pk / N_LAYERS, dma_avg_packet_bytes=sz / pk,
             hbm_read_mb_per_layer=sum(s["hbm_read_bytes"] for s in rs) / len(rs) / N_LAYERS / 1e6,
             spill_reload_bytes=sum(s["spill_reload_bytes"] for s in rs) / len(rs),
             spill_save_bytes=sum(s["spill_save_bytes"] for s in rs) / len(rs),
             cc_op_count=sum(s["cc_op_count"] for s in rs) / len(rs), n_ranks=len(allr))
    return r


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    res = {}
    for s in specs:
        label, rest = s.split("=", 1)
        neff, pat = rest.split(":", 1)
        res[label] = row(neff, sorted(glob.glob(pat)))
        print(label, json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in res[label].items()}),
              flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
