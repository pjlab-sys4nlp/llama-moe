"""
Create a LLaMA MoE model with SwitchBalancedGate.
"""

import argparse

import numpy as np
import torch.cuda
from transformers import LlamaTokenizer

from smoe.models.llama_moefication.configuration_llama_moe import LlamaMoEConfig
from smoe.models.llama_moefication.modeling_llama_moe import (
    LlamaMoEForCausalLM,
    LlamaMoEForSequenceClassification,
    LlamaMoEModel,
)


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    """set up configs"""
    # 模型大小参数
    intermediate_size = 11008
    num_hidden_layers = 32

    # MoE专家配置
    num_experts = 4
    num_selects = 1  # SwitchBalancedGate 的选择数量只能为1
    size_experts = []  # 每个专家拥有的神经元数量，如果为None则各个专家大小相同

    # MoE门控网络配置
    gate_type = "SoftMoEGate"
    slots_per_expert = 1

    # MoE计算方法配置
    calculator_type = "SoftMoECalculator"

    # 随机生成各个专家的大小，添加到size_experts
    for i in range(num_hidden_layers):
        this_size = np.random.randint(
            1, high=intermediate_size // num_experts + 1, size=num_experts
        )
        diff = intermediate_size - np.sum(this_size)  # 调整列表中的数字，使总和达到目标值
        this_size[-1] += diff
        size_experts.append(this_size)
    print("size_experts: ", size_experts)

    """create model"""
    print("Creating model...")
    config_llama_moe = LlamaMoEConfig(
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_experts=num_experts,
        num_selects=num_selects,
        size_experts=size_experts,
        gate_type=gate_type,
        slots_per_expert=slots_per_expert,
        calculator_type=calculator_type,
    )

    if args.model_type == "LlamaMoEModel":
        model_llama_moe = LlamaMoEModel(config_llama_moe)
    elif args.model_type == "LlamaMoEForCausalLM":
        model_llama_moe = LlamaMoEForCausalLM(config_llama_moe)
    elif args.model_type == "LlamaMoEForSequenceClassification":
        model_llama_moe = LlamaMoEForSequenceClassification(config_llama_moe)
    else:
        raise ValueError

    """prepare data"""
    sentence_list = [
        "hi hi hi hi hi, hi hi hi hi hi, hi hi hi hi hi",
        "How are you? I'm fine, and you?",
        "<s> <unk> <unk> <unk> <unk> <unk> </s>",
        "I am stupid. Are you sure?",
        "The past is never dead. It is not even past.",
    ]

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(sentence_list, padding=True, return_tensors="pt")
    print(tokens)

    """forward test"""
    print("Forwarding inputs...")
    model_llama_moe.to(device).half()
    tokens.to(device)
    result = model_llama_moe(**tokens)  # noqa: F841
    # print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=(
            "LlamaMoEModel",
            "LlamaMoEForCausalLM",
            "LlamaMoEForSequenceClassification",
        ),
    )
    args = parser.parse_args()
    main(args)
