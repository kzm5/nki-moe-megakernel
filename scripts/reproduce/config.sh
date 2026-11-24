# reproduce.sh configuration (sourced). Configs x lengths x n; paths.
CONFIGS="B C V"            # any of: A B C E Eh V   (Eh needs EH_DIR)
LENGTHS="768"              # multiples of 256 (LNC=2); 768 is the frozen validator's length
N=3                        # benchmark rounds (interleaved, one process per run)

VENV=$HOME/venv-neuron-229
MODEL_PATH=$HOME/qwen-30b-a3b/hf_model
BUILD_ROOT=$HOME/qwen-30b-a3b/repro      # new builds go to $BUILD_ROOT/<cfg>_s<len>
OUT_ROOT=$HOME/runs/repro                # validate/ bench/ outputs
P1=$HOME/runs/p1                         # frozen validator data: refs/, golden/, val/xla_plain (768 XLA baseline)
EH_DIR=                                  # worktree of branch q3/t3-fusion (Eh lives only there)

# Optional: reuse an existing build for one config/length, e.g.
# BUILD_B_768=$HOME/qwen-30b-a3b/traced_xlamoe_trn2_229_s768_greedy
