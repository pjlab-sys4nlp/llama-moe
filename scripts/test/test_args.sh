#!/usr/bin/bash

templates="1234567"

gpus=0
cpus=1
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
  python -m tests.utils.test_args \
  --t ${templates}
