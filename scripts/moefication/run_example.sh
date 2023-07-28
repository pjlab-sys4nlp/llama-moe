#!/usr/bin/bash

gpus=1
cpus=24
OMP_NUM_THREADS=8 srun --partition=MoE --job-name=test --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --job-name=example --kill-on-bad-exit=1 \
  python -m smoe.entrypoint.example_llama_moe
