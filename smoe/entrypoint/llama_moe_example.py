from transformers import LlamaTokenizer

from smoe.models.llama_moefication import LlamaMoEForCausalLM


def main():
    model_dir = (
        "/mnt/petrelfs/share_data/quxiaoye/models/llama_7B_MoE_16Select4-l2_norm"
    )

    # 读取现有模型
    print("Loading model...")
    model_llama_moe = LlamaMoEForCausalLM.from_pretrained(model_dir)
    # 可以修改专家数量
    model_llama_moe.change_num_selects(4)

    """prepare data"""
    sentence_list = [
        "hi hi hi hi hi, hi hi hi hi hi, hi hi hi hi hi",
        "How are you? I'm fine, and you?",
        "<s> <unk> <unk> <unk> <unk> <unk> </s>",
        "I am stupid. Are you sure?",
        "The past is never dead. It is not even past.",
    ]

    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(sentence_list, padding=True, return_tensors="pt")
    print(tokens)

    """forward test"""
    model_llama_moe.to("cuda:0")
    tokens.to("cuda:0")
    result = model_llama_moe(**tokens)
    print(result)


if __name__ == "__main__":
    main()
