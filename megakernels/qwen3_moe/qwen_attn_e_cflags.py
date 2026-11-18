# coding=utf-8
"""Config E compiled with the megakernel's (config C's) compiler flags instead of qwen.py's."""
from megakernels.qwen3_moe import qwen_attn_e
from megakernels.qwen3_moe.qwen_with_megakernel import NeuronQwen3MoeForCausalLMV3


class NeuronQwen3MoeForCausalLM(qwen_attn_e.NeuronQwen3MoeForCausalLM):
    def get_compiler_args(self):
        return NeuronQwen3MoeForCausalLMV3.get_compiler_args(self)
