"""Append one run to ~/runs/ledger.jsonl. Usage: ledger_add.py RUN_ID LOG REPORT_JSON GIT_SHA START_ISO ENABLE_NKI SEQ_LEN VENV"""
import json, re, sys, importlib.metadata as md
run_id, log, report, sha, start, nki, seq_len, venv = sys.argv[1:9]
text = open(log).read()
rep = json.load(open(report))
pkgs = ["neuronx-cc", "torch-neuronx", "torch", "torch-xla", "libneuronxla", "nki",
        "neuronx-distributed", "neuronx-distributed-inference", "transformers", "huggingface-hub"]
m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", text)
parts = [float(x) for x in m.group(1).split(":")]
wall = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
rss_kb = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", text).group(1))
entry = {
    "run_id": run_id,
    "timestamp": start,
    "status": "ok",
    "venv": venv,
    "config": {"backend": "NKI megakernel" if nki == "1" else "XLA", "platform_target": "trn2",
               "tp_degree": 4, "lnc": 2, "seq_len": int(seq_len), "batch_size": 1, "model": "Qwen3-30B-A3B"},
    "packages": {p: md.version(p) for p in pkgs},
    "git_sha": sha,
    "wall_seconds": round(wall, 2),
    "compile_seconds": (lambda m: float(m.group(1)) if m else None)(re.search(r"Compiling and tracing time: ([\d.]+)", text)),
    "tkg_latency_ms_p50": rep["token_generation_model"]["latency_ms_p50"],
    "tkg_latency_ms_p99": rep["token_generation_model"]["latency_ms_p99"],
    "e2e_throughput_tok_s": rep["e2e_model"]["throughput"],
    "peak_rss_gib": round(rss_kb / 2**20, 2),
    "log": log,
}
with open("/home/ubuntu/runs/ledger.jsonl", "a") as f:
    f.write(json.dumps(entry) + "\n")
print(json.dumps(entry, indent=1))
