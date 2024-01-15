"""
Load ReLU LLaMA model from file.
"""

import argparse

import torch.cuda
from transformers import LlamaForCausalLM, LlamaTokenizer

from smoe.utils.model_operation.modify_llama_model import llama_with_relu_activation


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")

    model = LlamaForCausalLM.from_pretrained(args.model_path)
    model.model = llama_with_relu_activation(model.model)
    model.config.use_cache = True

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
    result = model.generate(**tokens, repetition_penalty=2.0, max_length=256)
    print(result)

    for i in range(result.shape[0]):
        print(result[i])
        decoded_text = tokenizer.decode(result[i], skip_special_tokens=True)
        print(decoded_text)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/mnt/petrelfs/share_data/quxiaoye/models/ReluLLaMA-7B",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/petrelfs/share_data/quxiaoye/models/ReluLLaMA-7B",
    )
    args = parser.parse_args()
    main(args)
