#!/usr/bin/bash

llama_size="llama_7B" #  7B  13B  30B  base

num_experts=8
metric=l2 # l2 cos
template=layers.{}.mlp.up_proj.weight

root_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${root_path}/models/${llama_size}
save_path=${root_path}/moefication_results/split

gpus=0
cpus=32
OMP_NUM_THREADS=8 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
  python -m smoe.entrypoint.moefication.llama_split_clustering \
  --model_path ${model_path} \
  --save_path ${save_path} \
  --template ${template} \
  --num_experts ${num_experts} \
  --metric ${metric}
