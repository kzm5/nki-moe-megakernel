"""Benchmark one build with main.py's arguments and benchmark_sampling, without the trn3 evaluate_single
baseline compile (logit_validation).

Usage:
    bench_only.py [main.py args ...] --out report.json
"""
import importlib, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)


def main():
    out = sys.argv[sys.argv.index("--out") + 1]
    i = sys.argv.index("--out")
    sys.argv = sys.argv[:i] + sys.argv[i + 2:]
    import main as M
    args = M.parse_args()
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = args.platform_target
    if not args.prompts:
        args.prompts = ["I believe the meaning of life is"]
    args.batch_size = len(args.prompts)
    args.max_length = args.seq_len
    mod = "megakernels.qwen3_moe.qwen_with_megakernel" if args.enable_nki else (
        args.qwen if args.qwen != "qwen" else "megakernels.qwen3_moe.qwen")
    cls = getattr(importlib.import_module(mod), "NeuronQwen3MoeForCausalLM")
    args.draft_model_cls = None
    model, tok, gen_cfg = M.prepare_inference(cls, args)
    report = M.benchmark_sampling(model, tok, gen_cfg, args.prompts)
    json.dump(report, open(out, "w"), indent=1)
    print("BENCH_DONE", json.dumps(report.get("token_generation_model", {})))


if __name__ == "__main__":
    main()
