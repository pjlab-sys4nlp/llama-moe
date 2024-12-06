#!/usr/bin/bash

validation_dir=/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized
batch_size=4

gpus=1
cpus=16
quotatype=reserved # auto spot reserved

model_path=/mnt/petrelfs/share_data/quxiaoye/models/llama-moe-models/LLaMA-MoE-v1-3_0B-2_16
save_path=/mnt/petrelfs/dongdaize.d/workspace/llama-moe/visualization/gating-pair/2_16

OMP_NUM_THREADS=4 srun --partition=MoE --job-name=vis --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
  python -m smoe.entrypoint.analysis.gating_pair \
  --model_path ${model_path} \
  --validation_dir ${validation_dir} \
  --batch_size ${batch_size} \
  --save_path ${save_path}
