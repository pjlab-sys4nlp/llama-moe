#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_7B"

retain_percent=0.05 #  0.99 0.98 0.95 0.90 0.80 0.75 0.70 0.60 0.50 0.40 0.30 0.25 0.20 0.13 0.10 0.06 0.05
proj_type=up_proj   #  gate_proj  up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
save_path=${data_path}/moefication_results/prune

gpus=0
cpus=8
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.expert_construction.llama_prune_random \
  --model_path ${model_path} \
  --save_path ${save_path} \
  --retain_percent ${retain_percent} \
  --template layers.{}.mlp.${proj_type}.weight

chmod -R 755 ${save_path} >/dev/null 2>&1
