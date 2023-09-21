"""
Create a LLaMA MoE Residual model with TopKBalancedNoisyGate.
"""

import argparse

import numpy as np
import torch.cuda
from transformers import LlamaTokenizer

from smoe.models.llama_moe_residual.configuration_llama_moe_residual import (
    LlamaMoEResidualConfig,
)
from smoe.models.llama_moe_residual.modeling_llama_moe_residual import (
    LlamaMoEResidualForCausalLM,
    LlamaMoEResidualForSequenceClassification,
    LlamaMoEResidualModel,
)


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    """set up configs"""
    # 模型大小参数
    intermediate_size = 11008
    intermediate_size_moe = 9632  # 🔍
    intermediate_size_residual = 1376  # 🔍
    num_hidden_layers = 32

    # MoE专家配置
    num_experts = 14
    num_selects = 2
    size_experts = None  # 每个专家拥有的神经元数量，如果为None则各个专家大小相同

    # Residual模块配置 🔍
    num_experts_residual = 2
    size_experts_residual = None  # 688
    score_scale_factor_residual = 8.0
    use_weighting = False

    # MoE门控网络配置
    gate_type = "TopKBalancedNoisyGate"
    gate_network = "mlp"
    gate_use_softmax = True
    gate_use_balance = True
    gate_balance_loss_weight = 1e-2
    gate_add_noise = True
    gate_noise_epsilon = 1e-2

    # MoE计算方法配置
    calculator_type = "UniversalCalculator"
    multiply_gate_scores = True
    score_scale_factor = 8.0

    # 随机生成各个专家的大小，添加到size_experts
    # for i in range(num_hidden_layers):
    #     this_size = np.random.randint(
    #         1, high=intermediate_size // num_experts + 1, size=num_experts
    #     )
    #     diff = intermediate_size - np.sum(this_size)  # 调整列表中的数字，使总和达到目标值
    #     this_size[-1] += diff
    #     size_experts.append(this_size)
    # print("size_experts: ", size_experts)

    """create model"""
    print("Creating model...")
    config_llama_moe_residual = LlamaMoEResidualConfig(
        intermediate_size=intermediate_size,
        intermediate_size_moe=intermediate_size_moe,
        intermediate_size_residual=intermediate_size_residual,
        num_hidden_layers=num_hidden_layers,
        num_experts=num_experts,
        num_selects=num_selects,
        size_experts=size_experts,
        num_experts_residual=num_experts_residual,
        size_experts_residual=size_experts_residual,
        score_scale_factor_residual=score_scale_factor_residual,
        use_weighting=use_weighting,
        gate_type=gate_type,
        gate_network=gate_network,
        gate_use_softmax=gate_use_softmax,
        gate_use_balance=gate_use_balance,
        gate_balance_loss_weight=gate_balance_loss_weight,
        gate_add_noise=gate_add_noise,
        gate_noise_epsilon=gate_noise_epsilon,
        calculator_type=calculator_type,
        multiply_gate_scores=multiply_gate_scores,
        score_scale_factor=score_scale_factor,
    )

    if args.model_type == "LlamaMoEResidualModel":
        config_llama_moe_residual = LlamaMoEResidualModel(config_llama_moe_residual)
    elif args.model_type == "LlamaMoEResidualForCausalLM":
        config_llama_moe_residual = LlamaMoEResidualForCausalLM(
            config_llama_moe_residual
        )
    elif args.model_type == "LlamaMoEForSequenceClassification":
        config_llama_moe_residual = LlamaMoEResidualForSequenceClassification(
            config_llama_moe_residual
        )
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
    config_llama_moe_residual.to(device)
    tokens.to(device)
    result = config_llama_moe_residual(**tokens)  # noqa: F841
    # print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=(
            "LlamaMoEResidualModel",
            "LlamaMoEResidualForCausalLM",
            "LlamaMoEResidualForSequenceClassification",
        ),
    )
    args = parser.parse_args()
    main(args)
