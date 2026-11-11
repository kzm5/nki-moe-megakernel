"""Write weights/tp*_sharded_checkpoint.safetensors into an existing build. Usage: preshard.py KIND COMPILED_DIR SEQ_LEN"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase  # noqa: E402

kind, cdir, seq = sys.argv[1:4]
C.FLAGS = ["--seq-len", seq, "--on-device-sampling", "--top-k", "1", "--do-sample", "",
           "--output-logits", "--save-sharded-checkpoint", "--skip-compile"]


class Done(Exception):
    pass


def shard_instead_of_load(self, path, *a, **k):
    t0 = time.time()
    self.shard_weights(path)
    print(f"PRESHARD_DONE {path} {time.time() - t0:.0f}s", flush=True)
    raise Done


NeuronApplicationBase.load = shard_instead_of_load
try:
    C.load(kind, cdir)
except Done:
    pass
