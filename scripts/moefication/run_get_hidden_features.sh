#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama2_7B"
save_interval=1
batch_size=4
data_use_percent=0.01

proj_type=gate_proj #  gate_proj  up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
train_data_path=${data_path}/data/moefication_LLAMA_data
train_data_cache_path=${data_path}/data/moefication_LLAMA_data_cache
save_path=${data_path}/moefication_results/features

gpus=8
cpus=$((gpus * 16))
OMP_NUM_THREADS=8 srun --partition=MoE --job-name=get_features --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
  torchrun --nproc_per_node=${gpus} -m smoe.entrypoint.moefication.llama_get_hidden_features \
  --model_path ${model_path} \
  --train_data_path ${train_data_path} \
  --train_data_cache_path ${train_data_cache_path} \
  --save_path ${save_path} \
  --template layers.{}.mlp.${proj_type}.weight \
  --data_use_percent ${data_use_percent} \
  --save_interval ${save_interval} \
  --batch_size ${batch_size}

wait
chmod -R 777 ${save_path}/${llama_size}
