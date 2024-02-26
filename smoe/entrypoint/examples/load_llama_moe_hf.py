"""
Load LLaMA MoE model from file.
"""

import argparse

import torch.cuda
from transformers import LlamaTokenizer

from smoe.models.llama_moe.modeling_llama_moe_hf import LlamaMoEModel, LlamaMoEForCausalLM


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")

    if args.model_type == "LlamaMoEModel":
        model = LlamaMoEModel.from_pretrained(args.model_path)
    elif args.model_type == "LlamaMoEForCausalLM":
        model = LlamaMoEForCausalLM.from_pretrained(args.model_path)
    else:
        raise ValueError

    model.config.use_cache = False

    # set moe configs
    model.set_moe_num_selects(1)  # 修改专家的选择数量

    # set gate configs
    model.set_moe_gate_use_softmax(True)  # 修改是否使用Softmax对门控输出进行激活
    model.set_moe_gate_use_balance(True)  # 修改是否在训练时使用loss平衡专家选择的样本数量
    model.set_moe_gate_balance_loss_weight(0.02)  # 修改平衡loss的权重
    model.set_moe_gate_add_noise(True)  # 修改是否在训练时添加随机噪声到门控输出
    model.set_moe_gate_noise_epsilon(0.02)  # 修改噪声的大小

    # set calculator configs
    model.set_moe_calculator_multiply_gate_scores(True)  # 修改是否对专家输出加权
    model.set_moe_calculator_score_scale_factor(16.0)  # 修改专家输出的权重放缩倍数

    # reset
    model.reset_gate_network()  # 重新随机初始化门控网络
    model.reset_experts()  # 重新初始化专家参数

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
    model.half()
    model.to(device)
    tokens.to(device)
    result = model.generate(**tokens, repetition_penalty=1.3, max_length=256)
    print(result)

    for i in range(result.shape[0]):
        print(result[i])
        decoded_text = tokenizer.decode(result[i], skip_special_tokens=True)
        print(decoded_text)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--model_path", type=str)
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
