#!/usr/bin/bash

model_path=/mnt/petrelfs/share_data/quxiaoye/runs/llama2_random_scale4_112gpus_dynamic_data/latest.dir/checkpoint-13600
validation_dir=/mnt/petrelfs/share_data/quxiaoye/data/llama1_7B_val_set_tokenized
batch_size=8
save_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/analysis/

gpus=1
cpus=16
quotatype=auto # auto spot reserved
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=visualize --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
  python -m smoe.entrypoint.analysis.gating_pair \
  --model_path ${model_path} \
  --validation_dir ${validation_dir} \
  --batch_size ${batch_size} \
  --save_path ${save_path}
