#!/usr/bin/bash

tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/llama-moe-models/LLaMA-MoE-v1-3_0B-2_16
model_path=/mnt/petrelfs/share_data/quxiaoye/models/llama-moe-models/LLaMA-MoE-v1-3_0B-2_16

model_type=LlamaMoEForCausalLM #  LlamaMoEModel  LlamaMoEForCausalLM

gpus=1
cpus=8
OMP_NUM_THREADS=2 srun --partition=MoE --job-name="☝☝☝" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=spot \
  python -m smoe.entrypoint.examples.load_llama_moe_hf \
  --tokenizer_path ${tokenizer_path} \
  --model_path ${model_path} \
  --model_type ${model_type}
