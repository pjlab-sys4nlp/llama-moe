# train-moe

[[MoEfication Docs]](docs/moefication/README.md)

## 🌴 Dependencies

- Python >= 3.11
    - scikit-learn>=1.3.0
    - omegaconf>=2.0.6
    - tqdm>=4.65.0
    - datasets>=2.13.1
    - transformers>=4.30.2
    - peft>=0.4.0
    - xformers>=0.0.20
    - k_means_constrained==0.7.3
    - install flash-attention followed by this instruction: https://github.com/Dao-AILab/flash-attention

## 🚀 QuickStart

### Tokenization

- RedPajama: `bash scripts/tokenize/redpajama.sh` (Don't forget to change the folder paths.)

### Continual Pre-training (CPT)

**NOTICE:** Please create `logs/` folder manually: `mkdir -p logs`

- LLaMA MoEfication LoRA: `sbatch scripts/cpt/lora.sh`

## 🤝 Contribution

- Make sure the Python version `>=3.10` (a strict version contraint for better type hinting)

```bash
$ git clone git@github.com:pjlab-sys4nlp/train-moe.git
$ cd train-moe
$ pip install -e .[dev]
$ pre-commit install
```

## 🔗 Experiments

- CPT
  - [MoEfication L2-norm 8选4 继续预训练实验](https://m04hsypyylv.feishu.cn/docx/R9Tid61U0oOuQ4xwrbGcyCyvnMf)
