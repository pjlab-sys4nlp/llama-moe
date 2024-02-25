# Supervised Fine-Tuning (SFT)

## Data Preparation

Download [Deita 6K](https://huggingface.co/datasets/hkust-nlp/deita-6k-v0) to `data/deita/deita_6k.jsonl`.

## Training

Start training in Slurm clusters: `sbatch scripts/sft/2_8.sh`.

## Inference

```python
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from src.utils.conversation import Conversation

conv = Conversation()
conv.append_message("human", "Give me a three-day plan in Suzhou.")
conv.append_message("gpt", None)
prompt = conv.get_prompt()
print(prompt)
print(prompt[-1] == " ")

model_dir = "llama-moe/LLaMA-MoE-v1-3_5B-2_8-sft"

tok = AutoTokenizer.from_pretrained(model_dir)
m = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
m.eval()
m.cuda()

inputs = tok(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].cuda()

output = m.generate(input_ids, max_length=100, temperature=1.0, do_sample=True, use_cache=True)
response = tok.decode(output[0], skip_special_tokens=True)
print(response)
```
