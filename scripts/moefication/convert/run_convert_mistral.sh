#!/usr/bin/bash

model_path=/mnt/petrelfs/share_data/quxiaoye/models/Mistral-7B-v0.1
moe_config_path=/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1
split_file_path=/mnt/petrelfs/share_data/quxiaoye/moefication_results/split/Mistral-7B-v0.1-8Expert-Split-Random
save_path=/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1-Random-8Select2

template=layers.{}.mlp.up_proj.weight
num_experts=8
num_selects=2

gpus=0
cpus=8
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=convert --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.moefication.mistral_convert \
  --model_path ${model_path} \
  --moe_config_path ${moe_config_path} \
  --split_file_path ${split_file_path} \
  --save_path ${save_path} \
  --template ${template} \
  --num_experts ${num_experts} \
  --num_selects ${num_selects}

chmod -R 755 ${save_path} >/dev/null 2>&1
