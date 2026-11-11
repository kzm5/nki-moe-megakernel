"""Shared loader for the trn2 correctness check (one model per process)."""
import importlib, json, os, sys
REPO = os.path.expanduser("~/nki-moe-megakernel")
sys.path.insert(0, REPO)
os.chdir(REPO)
import main as M  # noqa: E402

FLAGS = ["--seq-len", "768", "--on-device-sampling", "--top-k", "1",
         "--do-sample", "", "--output-logits", "--save-sharded-checkpoint", "--skip-compile"]
KINDS = {  # kind -> (extra flags, module)
    "xla": ([], "megakernels.qwen3_moe.qwen"),
    "libmoe": (["--moe-fused-nki-kernel-enabled"], "megakernels.qwen3_moe.qwen"),
    "nki": (["--enable-nki"], "megakernels.qwen3_moe.qwen_with_megakernel"),
}
N_NEW = 128
REPO_TOL = "{None: (1e-5, 0.05), 1000: (1e-5, 0.03), 50: (1e-5, 0.03), 5: (1e-5, 0.03)}"
RTOL1E2 = "{None: (1e-5, 0.01), 1000: (1e-5, 0.01), 50: (1e-5, 0.01), 5: (1e-5, 0.01)}"


def load(kind, compiled_dir):
    extra, mod = KINDS[kind]
    sys.argv = ["main.py", "--model", "qwen3_moe", "--platform-target", "trn2",
                "--model-path", os.path.expanduser("~/qwen-30b-a3b/hf_model"),
                "--compiled-model-path", compiled_dir, "--tp-degree", "4",
                "--logical-nc-config", "2"] + FLAGS + extra
    args = M.parse_args()
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = args.platform_target
    args.prompts = ["x"]
    args.batch_size = 1
    args.max_length = args.seq_len
    cls = getattr(importlib.import_module(mod), "NeuronQwen3MoeForCausalLM")
    args.draft_model_cls = None
    model, tok, gen_cfg = M.prepare_inference(cls, args)
    assert model.neuron_config.output_logits, "build does not return logits"
    return model, tok, gen_cfg


def prompts():
    return json.load(open(os.path.expanduser("~/runs/val/prompts.json")))
