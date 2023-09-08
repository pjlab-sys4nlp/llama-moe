#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_7B"

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
train_data_path=${data_path}/data/moefication_LLAMA_data
train_data_cache_path=${data_path}/data/moefication_LLAMA_data_cache

gpus=0
cpus=16
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=datasets --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
  python -m smoe.entrypoint.moefication.llama_prepare_datasets \
  --model_path ${model_path} \
  --train_data_path ${train_data_path} \
  --train_data_cache_path ${train_data_cache_path}

wait
chmod -R 777 ${train_data_cache_path} >/dev/null 2>&1
