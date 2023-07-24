# train-moe

## For developers

- Make sure the Python version `>=3.10` (a strict version contraint for better type hinting)

```bash
$ git clone git@github.com:pjlab-sys4nlp/train-moe.git
$ pip install -e .[dev]
$ pre-commit install
```

## Tokenization

- RedPajama: `bash scripts/tokenize/redpajama.sh`
