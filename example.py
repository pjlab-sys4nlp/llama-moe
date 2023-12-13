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
