<div align="center">
  <h1>LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training</h1>
  <img src="docs/imgs/title-favicon.png" width="200" alt="LLaMA-MoE favicon" style="border-radius: 5%;"><br />
  <span style="color:red">📢 <strong><i>A SMALLER AFFORDABLE MoE MODEL FOR EVERYONE!!</i></strong></span>
  <div>
    <a href="https://huggingface.co/llama-moe" target="_blank">🤗 Model Weights</a> | <a href="docs/LLaMA_MoE.pdf" target="_blank">📃 Technical Report</a> | <a href="#quick-start">🚀 Quick Start</a><br />
    <a href="#installation">⚙️ Installation Guide</a> | <a href="#expert-construction">🚧 Expert Construction</a> | <a href="#continual-pretraining">🚅 Continual Pre-training</a> | <a href="#evaluation">💎 Evaluation</a>
  </div>
</div>

<h2 id="llama-moe">🎉 Introduction</h2>

LLaMA-MoE is a series of open-sourced Mixture-of-Expert (MoE) models based on [LLaMA](https://github.com/facebookresearch/llama) and [SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama).
We build LLaMA-MoE with the following two steps:
1. Partition LLaMA's FFNs into sparse experts and insert top-K gate for each layer of experts.
2. Continually pre-train the initialized MoE model with an optimized data sampling weights from [Sheared LLaMA](https://arxiv.org/abs/2310.06694) and filtered datasets from [SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama).

![MoE Routing](./docs/imgs/MoE-Routing.gif)

<h2 id="features">🔥 Features</h2>

1. **Lightweight Models**: The number of activated model parameters is only 3.0~3.5B, which is friendly for deployment and research usage.
2. **Multiple Expert Construction Methods**:
   1. Neuron-Independent: Random, Clustering, Co-activation Graph, Gradient ([Zhang et al., 2022](http://arxiv.org/abs/2110.01786), [Zuo et al., 2022](http://arxiv.org/abs/2204.07675))
   2. Neuron-Sharing: Inner, Inter (residual)
3. **Multiple MoE Gating Strategies**:
   1. TopK Noisy Gate ([Shazeer et al., 2017](http://arxiv.org/abs/1701.06538))
   2. Switch Gating ([Fedus et al., 2022](http://arxiv.org/abs/2101.03961))
4. **Fast Continual Pre-training**:
   1. FlashAttention-v2 integrated ([Dao, 2023](https://github.com/Dao-AILab/flash-attention))
   2. Fast streaming dataset loading
5. **Abundant Monitor Items**:
   1. Gate load, gate importance
   2. Loss on steps, loss on tokens, balance loss
   3. TGS (tokens/GPU/second), MFU (model FLOPs utilization)
   4. Other visualization utilities
6. **Dynamic Weight Sampling**:
   1. Self-defined static sampling weights
   2. Sheared LLaMA's dynamic batch loading ([Xia et al., 2023](http://arxiv.org/abs/2310.06694))


<h2 id="quick-start">🚀 QuickStart</h2>

```python
# python>=3.10

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model.to("cuda:0")

input_text = "Suzhou is famous of"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda:0")

pred = model.generate(**inputs, max_length=50, temperature=0.0)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three
```

<h2 id="installation">⚙️ Installation</h2>

1. Prepare conda environment: `conda create -n smoe python=3.11` (If your environment name is not `smoe`, you may need to change environment in launching scripts)
2. Add correct environment variables in `~/.bashrc` (`gcc` is set to newer version for installing `flash-attn`). e.g.:
    ```bash
    export PATH=/mnt/petrelfs/share/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PATH
    export PATH=/mnt/petrelfs/share/gcc-10.1.0/bin:$PATH
    export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc-10.1.0/lib64:$LD_LIBRARY_PATH
    ```
3. Take the variables into effect: `source ~/.bashrc`
4. Install PyTorch (CUDA-11.8): `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
5. Install dependencies: `pip install -r requirements.txt`
6. Install `flash-attn`: `pip install flash-attn==2.0.1 --no-build-isolation`. You may need to follow the [flash-attn installation instructions](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) to avoid some errors.
7. Install the latest Git: `conda install git`
8. Clone the repo: `git clone git@github.com:pjlab-sys4nlp/llama-moe.git` (If you don't setup the ssh key to GitHub, you may not able to clone through ssh. Check the [docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) about it.)
9. Change current directory: `cd llama-moe`
10. Install `smoe` in [editable mode](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-e): `pip install -e .[dev]`
11. Setup `pre-commit` hooks: `pre-commit install`

<h2 id="performance">📊 Model Performance</h2>

| Model                     | \#Activated Experts | \#Experts | \#Activated Params |                                   Links                                   |
| :------------------------ | :-----------------: | :-------: | :----------------: | :-----------------------------------------------------------------------: |
| **LLaMA-MoE-3.0B**        |          2          |    16     |        3.0B        | [[🤗 HF Weights]](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_0B-2_16) |
| **LLaMA-MoE-3.5B (4/16)** |          4          |    16     |        3.5B        | [[🤗 HF Weights]](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-4_16) |
| **LLaMA-MoE-3.5B (2/8)**  |          2          |     8     |        3.5B        | [[🤗 HF Weights]](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-2_8)  |

| Model                                                                                 | Average  |   SciQ   |   PIQA   | WinoGrande |  ARC-e   | ARC-c (25) | HellaSwag (10) |  LogiQA  | BoolQ (32) | LAMBADA  | NQ (32)  | MMNLU (5) |
| :------------------------------------------------------------------------------------ | :------: | :------: | :------: | :--------: | :------: | :--------: | :------------: | :------: | :--------: | :------: | :------: | :-------: |
| [OPT-2.7B](https://huggingface.co/facebook/opt-2.7b)                                  |   50.3   |   78.9   |   74.8   |    60.8    |   54.4   |    34.0    |      61.4      |   25.8   |    63.3    |   63.6   |   10.7   |   25.8    |
| [Pythia-2.8B](https://huggingface.co/EleutherAI/pythia-2.8b)                          |   51.5   |   83.2   |   73.6   |    59.6    |   58.8   |    36.7    |      60.7      |   28.1   |    65.9    |   64.6   |   8.7    |   26.8    |
| [INCITE-BASE-3B](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1) |   53.7   |   85.6   |   73.9   |    63.5    |   61.7   |    40.3    |      64.7      |   27.5   |    65.8    |   65.4   |   15.2   |   27.2    |
| [Open-LLaMA-3B-v2](https://huggingface.co/openlm-research/open_llama_3b_v2)           |   55.6   |   88.0   |   77.9   |    63.1    |   63.3   |    40.1    |      71.4      |   28.1   |    69.2    |   67.4   |   16.0   |   26.8    |
| [Sheared-LLaMA-2.7B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-2.7B)         |   56.4   |   87.5   |   76.9   |    65.0    |   63.3   |    41.6    |      71.0      |   28.3   |    73.6    |   68.3   |   17.6   | **27.3**  |
| **LLaMA-MoE-3.0B**                                                                    |   55.5   |   84.2   |   77.5   |    63.6    |   60.2   |    40.9    |      70.8      | **30.6** |    71.9    |   66.6   |   17.0   |   26.8    |
| **LLaMA-MoE-3.5B (4/16)**                                                             | **57.7** |   87.6   | **77.9** |    65.5    | **65.6** |  **44.2**  |    **73.3**    |   29.7   |  **75.0**  | **69.5** | **20.3** |   26.8    |
| **LLaMA-MoE-3.5B (2/8)**                                                              |   57.6   | **88.4** |   77.6   |  **66.7**  |   65.3   |    43.1    |    **73.3**    |   29.6   |    73.9    |   69.4   |   19.8   |   27.0    |


<h2 id="expert-construction">🚧 Expert Construction</h2>

- Neuron-Independent
  - Independent<sub>Random</sub>: `bash ./scripts/expert_construction/split/run_split_random.sh`
  - Independent<sub>Clustering</sub>: `bash ./scripts/expert_construction/split/run_split_clustering.sh`
- Neuron-Sharing
  - Sharing<sub>Inner</sub>: `bash ./scripts/expert_construction/split/run_split_gradient.sh`
  - Sharing<sub>Inter</sub>: `bash ./scripts/expert_construction/split/run_split_gradient_residual.sh`

For more information, please refer to [Expert Construction docs](docs/expert_construction/README.md).

<h2 id="continual-pretraining">🚅 Continual Pre-training</h2>


### Tokenization

Download [SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) into `/path_to_data` and put data from different domains into separate folders:
  - `/path_to_data/en_arxiv`
  - `/path_to_data/en_book`
  - `/path_to_data/en_c4`
  - `/path_to_data/en_cc`
  - `/path_to_data/en_stack`
  - `/path_to_data/en_wikipedia`
  - `/path_to_data/github`

Each file should be end with `*.jsonl` and each line looks like:
```
{"id": "id-info", "content": "raw text to be tokenized"}
```

Run the following command to tokenize the data in each folder:

```bash
python -m smoe.utils.tokenize \
  -f jsonl \
  -t /path_to_tokenizer \
  -i /path_to_data/en_arxiv \
  -o /path_to_data_tokenized/en_arxiv
```

### Continual Pre-training (CPT)

- **NOTICE:** Please create `logs/` folder manually: `mkdir -p logs`
- To run the continual pre-training, please check the [CPT docs](docs/continual_pretraining/README.md).

<h2 id="evaluation">💎 Evaluation</h2>

- For evalution on Natural Questions (NQ), please refer to [opencompass](https://github.com/Spico197/opencompass/tree/main).
- For other tasks, please refer to [lm-eval-harness](https://github.com/spico197/smoe-eval).

<h2 id="citation">📑 Citation</h2>

```bibtex
@article{llama-moe-2023,
  title={LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training},
  author={LLaMA-MoE Team},
  journal={arXiv preprint arXiv:},
  url={https://arxiv.org/abs/},
  year={2023}
}
```

<hr>
<p align="center">LLaMA-MoE Team w/ ❤️</p>
