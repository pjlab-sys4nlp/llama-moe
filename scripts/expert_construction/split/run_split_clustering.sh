#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_13B"

num_experts=8    #  8  16
metric=l2         #  l2  cos
proj_type=up_proj #  gate_proj  up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
save_path=${data_path}/moefication_results/split

gpus=0
cpus=32
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.expert_construction.llama_split_clustering \
  --model_path ${model_path} \
  --save_path ${save_path} \
  --template layers.{}.mlp.${proj_type}.weight \
  --num_experts ${num_experts} \
  --metric ${metric} \
  --cpu_threads ${cpus}

chmod -R 755 ${save_path} >/dev/null 2>&1
