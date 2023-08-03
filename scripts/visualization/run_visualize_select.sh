#!/usr/bin/bash

llama_size="llama_7B"    #  7B  13B  30B  base
num_experts=16           #  8  16
split_type=Random #  Clustering-l2  Clustering-cos  Random
select_type=l2_norm      #  plain  positive  l2_norm

result_path=/mnt/petrelfs/share_data/quxiaoye/moefication_results/select/${split_type}/${llama_size}-${num_experts}Expert-Select-MLP-${select_type}
#result_path=/mnt/petrelfs/share_data/quxiaoye/moefication_results/select-up-direct/llama_7B-8Expert-Select-MLP-l2_norm
save_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/${split_type}/${llama_size}-${num_experts}Expert-Select-MLP-${select_type}
#save_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization-up-direct/llama_7B-8Expert-Select-MLP-l2_norm

gpus=0
cpus=4
OMP_NUM_THREADS=8 srun --partition=MoE --job-name=visualize --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
  python -m smoe.entrypoint.visualization.visualize_expert_select \
  --result_path ${result_path} \
  --save_path ${save_path}
