# train-moe

[[Installation Guide]](docs/Installation.md) | [[MoEfication Docs]](docs/moefication/README.md) | [[Continual Pre-training Docs]](docs/continual_pretraining/README.md)

## 🌴 Dependencies

- Python==3.11.4
  - Packages: please check `requirements.txt` (NOTE: `flash-attn` must be properly installed by following [their instructions](https://github.com/Dao-AILab/flash-attention))

## 🚀 QuickStart

### Tokenization

- RedPajama: `bash scripts/tokenize/redpajama.sh` (Don't forget to change the folder paths.)

### Continual Pre-training (CPT)

**NOTICE:** Please create `logs/` folder manually: `mkdir -p logs`

- LLaMA MoEfication LoRA: `sbatch scripts/cpt/lora.sh`
- LLaMA MoEfication Full-Parameter: `sbatch scripts/cpt/fpt.sh`

## 🤝 Contribution

- Make sure the Python version `>=3.10` (a strict version contraint for better type hinting)

```bash
$ conda install git  # upgrade git
$ git clone git@github.com:pjlab-sys4nlp/train-moe.git
$ cd train-moe
$ pip install -e .[dev]
$ pre-commit install
```
