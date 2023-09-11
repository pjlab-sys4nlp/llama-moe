import argparse

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--templates', type=str, default='layers.{}.mlp.gate_proj.weight')
    args = parser.parse_args()

    print(args.templates)
