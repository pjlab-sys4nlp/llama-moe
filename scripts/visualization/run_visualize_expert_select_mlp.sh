#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama2_7B"

num_experts=8            #  8  16
num_selects=2            #  2  4
split_type=Clustering-l2 #  Graph-l1_norm  Graph-l2_norm  Clustering-l2  Clustering-cos  Random
select_type=l2_norm      #  plain  positive  l1_norm  l2_norm
proj_type=gate_proj      #  up_proj  gate_proj

result_path=/mnt/petrelfs/share_data/quxiaoye/moefication_results/select/${split_type}/${llama_size}-${num_experts}Expert-Select-MLP-${select_type}
save_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/expert-select/${split_type}-${select_type}/${llama_size}-${num_experts}Select${num_selects}-${proj_type}

gpus=0
cpus=4
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=visualize --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.visualization.visualize_expert_select_mlp \
  --result_path ${result_path} \
  --save_path ${save_path} \
  --proj_type ${proj_type}
