"""Per-phase attribution of one NTFF's device time (neuron-profile per-instruction JSON), per decoder layer.

Computes:
    Each instruction is anchored to a phase by (source file, enclosing function); instructions from shared helpers,
    XLA ops or with no source take the phase of the nearest anchored instruction in time. DMA records (no source)
    are placed the same way by start time. Per phase, divided by --layers:
        wall_us     fair-share wall time: every instant with k phases active is split 1/k among them
        busy_us     summed instruction duration per engine (engine-local; engines overlap)
        dma_us      union of DMA transfer intervals; dma_mb bytes moved; stall_us summed evt_wait_time
Returns:
    JSON {phase: {...}} and a printed table. Phases are prefixed by stage: attn, moe, xla.

Usage:
    phase_attrib.py PROF.json --out OUT.json [--layers 48] [--src-root REPO]
"""
import argparse, ast, bisect, collections, json, os

ATTN = {
    "attn_e_nki.py": "attn.x_load_y_store",
    "attn_e_hoist_nki.py": "attn.x_load_y_store",
    "qkv_tkg.py": "attn.qkv",
    "rope.py": "attn.qk_rope",
    "attention_tkg.py": "attn.core",
    "cascaded_max.py": "attn.core",
    "cascaded_max_utils.py": "attn.core",
    "gen_mask_tkg.py": "attn.core",
    "output_projection_tkg.py": "attn.o_proj",
    "transformer_tkg.py": "attn.allreduce_gather",
}
FUNCS = {
    "transformer_qwen3_moe_speculative.py": {
        "_build_full_mask_hbm": "attn.mask_build", "_stream_shuffle_broadcast": "attn.mask_build",
        "_build_permuted_rope_hbm_from_pos": "attn.rope_build",
        "_store_shard_interleaved_sb_to_hbm": "attn.x_load_y_store", None: "mk.body"},
    "attention_block_tkg.py": {
        "_rms_norm_inplace": "attn.qk_norm", "_kv_cache_update": "attn.kv_update",
        "_update_flat_cache_batched": "attn.kv_update", "_update_flat_cache": "attn.kv_update",
        "_update_block_cache": "attn.kv_update", "_update_block_cache_batched": "attn.kv_update",
        "_update_block_cache_scalar": "attn.kv_update", None: "attn.block_other"},
}
MOE = {
    "router_topk.py": "moe.router", "mlp_tkg_gate_up_projection.py": "moe.gate_up",
    "mlp_tkg_down_projection.py": "moe.down", "selective_expert_impl.py": "moe.other", "moe_tkg.py": "moe.other",
    "moe_tkg_utils.py": "moe.other", "moe_block_tkg.py": "moe.other", "mlp_tkg_utils.py": "moe.other",
    "moe_fused_nki.py": "moe.other",
}
XLA_FILES = {"module.py", "model_base.py", "ops.py", "functional.py"}


def func_index(path):
    """Sorted [(start, end, name)] of top-level functions in a source file."""
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError):
        return []
    return sorted((n.lineno, n.end_lineno, n.name) for n in tree.body if isinstance(n, ast.FunctionDef))


def resolver(src_root):
    cache = {}

    def fn(src):
        path, _, line = src.rpartition(":")
        if path not in cache:
            p = path if os.path.exists(path) else None
            if p is None and src_root:
                for rel in ("megakernels/qwen3_moe/", "nki_kernels/_legacy/attention/", "nki_kernels/_legacy/moe/"):
                    c = os.path.join(src_root, rel, os.path.basename(path))
                    if os.path.exists(c):
                        p = c
            cache[path] = func_index(p) if p else []
        ln = int(line) if line.isdigit() else -1
        for a, b, name in cache[path]:
            if a <= ln <= b:
                return name
        return None
    return fn


def anchor(i, func_of):
    if i.get("label", "").startswith("CC-core"):
        return "cc"
    src = i.get("nki_source_location") or i.get("bir_debug_info_source_location") or ""
    f = os.path.basename(src.rpartition(":")[0])
    if f in ATTN:
        return ATTN[f]
    if f in MOE:
        return MOE[f]
    if f in FUNCS:
        m = FUNCS[f]
        return m.get(func_of(src), m[None])
    if f in XLA_FILES or (not src and i.get("hlo_name")):
        return "xla.ops"
    return None  # shared helper / no source: nearest anchor


def merged(iv):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + (cur_e - cur_s if cur_e is not None else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prof")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--src-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    d = json.load(open(a.prof))
    func_of = resolver(a.src_root)
    ins = sorted(d["instruction"], key=lambda i: i["timestamp"])
    tags = [anchor(i, func_of) for i in ins]
    at = [(i["timestamp"], t) for i, t in zip(ins, tags) if t and t != "cc"]
    ts = [x for x, _ in at]

    def nearest(t):
        k = bisect.bisect_left(ts, t)
        c = [j for j in (k - 1, k) if 0 <= j < len(ts)]
        return at[min(c, key=lambda j: abs(ts[j] - t))][1]

    stage_of_cc = {}
    for n, (i, t) in enumerate(zip(ins, tags)):
        if t is None:
            tags[n] = nearest(i["timestamp"])
        elif t == "cc":
            st = nearest(i["timestamp"]).split(".")[0]
            tags[n] = f"{st}.cc"
    L = a.layers
    ph = collections.defaultdict(lambda: {"busy_us": collections.Counter(), "n": 0, "stall_us": 0.0,
                                          "dma_mb": 0.0, "iv": [], "div": []})
    events = []
    for i, t in zip(ins, tags):
        p = ph[t]
        dur = i.get("duration", 0)
        p["busy_us"][i.get("label", "?")] += dur / 1e3 / L
        p["n"] += 1
        p["stall_us"] += i.get("evt_wait_time", 0) / 1e3 / L
        p["iv"].append((i["timestamp"], i["timestamp"] + dur))
        events += [(i["timestamp"], 1, t), (i["timestamp"] + dur, -1, t)]
    for x in d["dma"]:
        t = nearest(x["timestamp"])
        ph[t]["dma_mb"] += x.get("transfer_size", 0) / 1e6 / L
        ph[t]["div"].append((x["timestamp"], x["timestamp"] + x.get("duration", 0)))
    events.sort()
    dma_iv = sorted((x["timestamp"], x["timestamp"] + x.get("duration", 0)) for x in d["dma"])
    dma_starts = [s for s, _ in dma_iv]
    dma_end_max, m = [], 0
    for _, e in dma_iv:
        m = max(m, e)
        dma_end_max.append(m)

    def dma_overlap(s, e):
        """Time in [s, e) covered by DMA transfers (merged)."""
        k = bisect.bisect_right(dma_starts, e)
        iv = [(max(a, s), min(b, e)) for a, b in dma_iv[max(0, bisect.bisect_left(dma_end_max, s)):k] if b > s]
        return merged([x for x in iv if x[1] > x[0]])

    active, last, wall = collections.Counter(), None, collections.Counter()
    prev_t, gaps = None, collections.Counter()
    for tm, kind, t in events:
        if last is not None and tm > last:
            live = [k for k, v in active.items() if v > 0]
            for k in live:
                wall[k] += (tm - last) / len(live)
            if not live:
                ov = dma_overlap(last, tm)
                wall["idle"] += tm - last
                wall["idle_dma"] += ov
                gaps[(prev_t, t)] += tm - last
        active[t] += kind
        if kind == -1:
            prev_t = t
        last = tm
    res = {}
    for t, p in ph.items():
        res[t] = {"wall_us": wall[t] / 1e3 / L, "union_us": merged(p["iv"]) / 1e3 / L,
                  "dma_us": merged(p["div"]) / 1e3 / L, "dma_mb": p["dma_mb"], "stall_us": p["stall_us"],
                  "n_per_layer": p["n"] / L, "busy_us": {k: round(v, 2) for k, v in p["busy_us"].items()}}
    res["idle"] = {"wall_us": wall["idle"] / 1e3 / L, "dma_active_us": wall["idle_dma"] / 1e3 / L,
                   "by_boundary_us": {f"{a} -> {b}": round(v / 1e3 / L, 2) for (a, b), v in gaps.most_common(15)}}
    end = max(e for p in ph.values() for _, e in p["iv"])
    res["_total"] = {"wall_us": end / 1e3 / L, "n_instr": len(ins), "n_dma": len(d["dma"])}
    json.dump(res, open(a.out, "w"), indent=1)
    eng = ("Tensor", "Vector", "Scalar", "GpSimd", "Sync")
    print(f"{'phase':26s} {'wall':>7s} {'union':>7s} {'dma_us':>7s} {'dma_MB':>7s} {'n':>6s} " +
          " ".join(f"{e[:6]:>6s}" for e in eng))
    for t in sorted(res, key=lambda k: -res[k]["wall_us"]):
        r = res[t]
        if "busy_us" not in r:
            print(f"{t:26s} {r['wall_us']:7.1f}" + (f"  (DMA active {r['dma_active_us']:.1f})" if t == "idle" else ""))
            for k, v in r.get("by_boundary_us", {}).items():
                print(f"    idle {v:6.2f}  {k}")
            continue
        print(f"{t:26s} {r['wall_us']:7.1f} {r['union_us']:7.1f} {r['dma_us']:7.1f} {r['dma_mb']:7.3f} "
              f"{r['n_per_layer']:6.0f} " + " ".join(f"{r['busy_us'].get(e, 0):6.1f}" for e in eng))


if __name__ == "__main__":
    main()
