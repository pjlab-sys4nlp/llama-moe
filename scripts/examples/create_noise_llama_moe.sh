#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
base_model=llama_7B

model_type=LlamaMoEForCausalLM #  LlamaMoEModel  LlamaMoEForCausalLM  LlamaMoEForSequenceClassification

tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/${base_model}/

gpus=0
cpus=16
OMP_NUM_THREADS=8 srun --partition=MoE --job-name=test --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --job-name=example --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.examples.create_noise_llama_moe \
  --tokenizer_path ${tokenizer_path} \
  --model_type ${model_type}
