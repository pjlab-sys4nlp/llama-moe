import argparse
import os

from tqdm import tqdm
from transformers import LlamaConfig

from smoe.utils.moefication.prune_llama import RandomPrune

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--retain_percent', type=float)
    parser.add_argument('--template', type=str, default='layers.{}.mlp.gate_proj.weight')

    args = parser.parse_args()
    print(args, "\n")

    print("Loading llama config...")
    config = LlamaConfig.from_pretrained(args.model_path)
    expert_size = int(config.intermediate_size * args.retain_percent)

    args.save_path = os.path.join(
        args.save_path,
        f"{os.path.split(args.model_path)[1]}-Prune-Random",
        f"{format(args.retain_percent, '.2f')}Percent-{expert_size}Neurons"
    )

    for i in tqdm(range(config.num_hidden_layers)):
        split = RandomPrune(args, args.template, i, config.intermediate_size)
        split.prune(expert_size, seed=0)
        split.save()
    print("Done.")
    # fmt: on
