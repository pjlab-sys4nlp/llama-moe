import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizer

from huggingface_hub import snapshot_download

repo_to_download = "openlm-research/open_llama_3b"
target_dir = "/mnt/petrelfs/share_data/quxiaoye/models/llama_3B"

snapshot_download(repo_id=repo_to_download, local_dir=target_dir, local_dir_use_symlinks=False)

tokenizer = LlamaTokenizer.from_pretrained(target_dir)
model = LlamaForCausalLM.from_pretrained(
    target_dir, torch_dtype=torch.float16, device_map='cpu',
)

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
)
print(tokenizer.decode(generation_output[0]))
