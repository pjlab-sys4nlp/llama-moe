<div align="center">
  <h1>LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training</h1>
  <img src="docs/imgs/title-favicon.png" width="200" alt="LLaMA-MoE favicon" style="border-radius: 5%;"><br />
  <span style="color:red">📢 <i>A SMALLER AFFORDABLE MoE MODEL FOR EVERYONE!!</i></span>
  <div>
    <a href="#" target="_blank">🤗 Model Weights</a> | <a href="#" target="_blank">📃 Technical Report</a> | <a href="#quick-start">🚀 Quick Start</a><br />
    <a href="docs/Installation.md">⚙️ Installation Guide</a> | <a href="#expert-initialization">🚧 Expert Initialization</a> | <a href="#continual-pretraining">🚅 Continual Pre-training</a> | <a href="#evaluation">💎 Evaluation</a>
  </div>
</div>

<h2 id="llama-moe">🎉 Introduction</h2>

LLaMA-MoE is a series of Mixture-of-Expert (MoE) models based on [LLaMA](https://github.com/facebookresearch/llama).
We build LLaMA-MoE with the following two steps:
1. Partition LLaMA's FFNs into sparse experts and insert top-K gate for each layer of experts.
2. Continually pre-train the initialized MoE model with an optimized data sampling weights from [Sheared LLaMA](https://arxiv.org/abs/2310.06694) and filtered datasets from [SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama).


| Model              | \#Activated Experts | \#Experts | \#Activated Params | \#Total Prams |                                              Links                                               |
| :----------------- | :-----------------: | :-------: | :----------------: | :-----------: | :----------------------------------------------------------------------------------------------: |
| OPT-2.7B           |          -          |     -     |        2.7B        |     2.7B      |                 ([Zhang et al., 2022](https://huggingface.co/facebook/opt-2.7b))                 |
| Pythia-2.8B        |          -          |     -     |        2.8B        |     2.8B      |             ([Biderman et al., 2023](https://huggingface.co/EleutherAI/pythia-2.8b))             |
| INCITE-BASE-3B     |          -          |     -     |        2.8B        |     2.8B      | ([Together Computer, 2023](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1)) |
| Open-LLaMA-3B-v2   |          -          |     -     |        3.4B        |     3.4B      |          ([Geng et al., 2023](https://huggingface.co/openlm-research/open_llama_3b_v2))          |
| Sheared-LLaMA-2.7B |          -          |     -     |        2.7B        |     2.7B      |          ([Xia et al., 2023](https://huggingface.co/princeton-nlp/Sheared-LLaMA-2.7B))           |
| **LLaMA-MoE-3.0B** |          2          |    16     |        3.0B        |     6.7B      |                              [[HF Weights]](https://huggingface.co)                              |
| **LLaMA-MoE-3.5B** |          4          |    16     |        3.5B        |     6.7B      |                              [[HF Weights]](https://huggingface.co)                              |



| Model              |   SciQ   |   PIQA   | WinoGrande |  ARC-e   | ARC-c (25) | HellaSwag (10) |  LogiQA  | BoolQ (32) | LAMBADA  | NQ (32)  | MMNLU (5) |  Average  |
| :----------------- | :------: | :------: | :--------: | :------: | :--------: | :------------: | :------: | :--------: | :------: | :------: | :-------: | :-------: |
| OPT-2.7B           |   78.9   |   74.8   |    60.8    |   54.4   |    34.0    |      61.4      |   25.8   |    63.3    |   55.9   |   10.7   |   25.8    |   49.6    |
| Pythia-2.8B        |   83.2   |   73.6   |    59.6    |   58.8   |    36.7    |      60.7      |   28.1   |    65.9    |   54.4   |   8.6    |   26.8    |   50.6    |
| INCITE-BASE-3B     |   85.6   |   73.9   |    63.5    |   61.7   |    40.3    |      64.7      |   27.5   |    65.8    |   55.6   |   15.2   |   27.2    |   52.8    |
| Open-LLaMA-3B-v2   | **88.0** |   77.9   |    63.1    |   63.3   |    40.1    |      71.4      |   28.1   |    69.2    |   59.5   |   16.0   |   26.8    |   54.9    |
| Sheared-LLaMA-2.7B |   87.5   |   76.9   |    65.0    |   63.3   |    41.6    |      71.0      |   28.3   |    73.6    |   59.7   |   17.7   | **27.3**  |   55.6    |
| **LLaMA-MoE-3.0B** |   84.2   |   77.5   |    63.6    |   60.2   |    40.9    |      70.8      | **30.6** |    71.9    |   59.7   |   17.0   |   26.8    |   54.8    |
| **LLaMA-MoE-3.5B** |   87.6   | **77.9** |  **65.5**  | **65.6** |  **44.2**  |    **73.3**    |   29.7   |  **75.0**  | **63.2** | **20.3** |   26.8    | **57.2 ** |


<h2 id="quick-start">🚀 QuickStart</h2>

```python
import torch
from transformers import AutoTokenizer
from smoe.models.llama_moe import LlamaMoEForCausalLM


model_dir = "/mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_split_112gpus_16_2/outputs/cpt-llama2_random_split_112gpus_16_2_scale_factor_8-2342244/checkpoint-13600/"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = LlamaMoEForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
model.to("cuda:0")

input_text = "Suzhou is famous of"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda:0")

pred = model.generate(**inputs, max_length=50, temperature=0.0)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three
```

<h2 id="expert-initialization">🚧 Expert Initialization</h2>

- Neuron-Independent
  - Independent<sub>Random</sub>: `bash ./scripts/moefication/split/run_split_random.sh`
  - Independent<sub>Clustering</sub>: `bash ./scripts/moefication/split/run_split_clustering.sh`
- Neuron-Sharing
  - Sharing<sub>Inner</sub>: `bash ./scripts/moefication/split/run_split_gradient.sh`
  - Sharing<sub>Inter</sub>: `bash ./scripts/moefication/split/run_split_gradient_residual.sh`

For more information, please refer to [Expert Initialization docs](docs/moefication/README.md).

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
