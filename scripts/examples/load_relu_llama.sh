#!/usr/bin/bash

base_model=ReluLLaMA-7B
model_path=/mnt/petrelfs/share_data/quxiaoye/models/${base_model}/

gpus=1
cpus=8
quotatype=spot # spot reserved auto
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=example --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --job-name=example --kill-on-bad-exit=1 --quotatype=${quotatype} \
  python -m smoe.entrypoint.examples.load_relu_llama \
  --tokenizer_path ${model_path} \
  --model_path ${m
