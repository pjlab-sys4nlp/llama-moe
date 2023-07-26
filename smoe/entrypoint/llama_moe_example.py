import numpy as np
from transformers import LlamaTokenizer

from smoe.models.llama_moefication.configuration_llama_moe import LlamaMoEConfig
from smoe.models.llama_moefication.modeling_llama_moe import LlamaMoEForCausalLM


def main():
    device = "cuda:0"
    tokenizer_path = "/mnt/petrelfs/share_data/quxiaoye/models/llama_7B"

    load_from_file = False
    model_path = (
        "/mnt/petrelfs/share_data/quxiaoye/models/llama_7B_MoE_8Select2-l2_norm"
    )

    if load_from_file:  # 从文件读取现有模型
        print("Loading model...")
        model_llama_moe = LlamaMoEForCausalLM.from_pretrained(model_path)
        model_llama_moe.change_moe_num_selects(4)  # 修改专家的选择数量
        model_llama_moe.change_moe_gate_add_noise(True)  # 修改是否在训练时添加随机噪声到门控输出
        model_llama_moe.change_moe_gate_use_balance(True)  # 修改是否在训练时使用loss平衡专家选择的样本数量
        model_llama_moe.change_moe_gate_use_softmax(True)  # 修改是否使用Softmax对门控输出进行激活

    else:  # 从零创建模型
        """randomly generate intermediate sizes for experts"""
        intermediate_size = 11008
        num_hidden_layers = 32
        num_experts = 16
        num_selects = 8
        size_experts = []  # 每个专家拥有的神经元数量，如果为None则各个专家大小相同

        for i in range(num_hidden_layers):
            this_size = np.random.randint(
                1, high=intermediate_size // num_experts + 1, size=num_experts
            )  # 生成随机列表
            diff = intermediate_size - np.sum(this_size)  # 调整列表中的数字，使总和达到目标值
            this_size[-1] += diff
            size_experts.append(this_size)
            print(this_size)

        """create model"""
        print("Creating model...")
        config_llama_moe = LlamaMoEConfig(
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_experts=num_experts,
            num_selects=num_selects,
            size_experts=size_experts,
        )
        model_llama_moe = LlamaMoEForCausalLM(config_llama_moe)
        # model_llama_moe = LlamaMoEForSequenceClassification(config_llama_moe)

    print(model_llama_moe)

    """prepare data"""
    sentence_list = [
        "hi hi hi hi hi, hi hi hi hi hi, hi hi hi hi hi",
        "How are you? I'm fine, and you?",
        "<s> <unk> <unk> <unk> <unk> <unk> </s>",
        "I am stupid. Are you sure?",
        "The past is never dead. It is not even past.",
    ]

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(sentence_list, padding=True, return_tensors="pt")
    print(tokens)

    """forward test"""
    model_llama_moe.to(device)
    tokens.to(device)
    result = model_llama_moe(**tokens)
    print(result)


if __name__ == "__main__":
    main()
