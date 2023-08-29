#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base

llama_size="llama_7B"

reinit_gate=True
data_begin_index=200
data_end_index=400
batch_size=8

share_path=/mnt/petrelfs/share_data/quxiaoye
tokenizer_path=${share_path}/models/${llama_size}

#model_path=${share_path}/models/tzhu_model_bak/cpt-moe-fpt-64gpus-bs16_2-zero1default-1600316/checkpoint-23000
model_path=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B_MoE_16Select4-l2_norm
#model_path=/mnt/petrelfs/zhutong/smoe/outputs/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772/checkpoint-8000
#model_path=${share_path}/models/LlamaMoEForCausalLM/Random-l2_norm/llama2_7B-16Select4-up_proj

#  commoncrawl-part-000203-16de0c55-head1000.jsonl
#  github-part-003227-16de0c55-head1000.jsonl
#  wikipedia-part-003428-16de0c55-head1000.jsonl
data_name=commoncrawl-part-000203-16de0c55-head1000.jsonl
data_path=${share_path}/data/vis_data/${data_name}

#data_path=/mnt/petrelfs/share_data/quxiaoye/part-003190-16de0c55.jsonl

#data_path=/mnt/petrelfs/share_data/quxiaoye/data/vis_data/head30_shuffled_output/shuffled_20.txt

save_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/expert-load-0B-Ini/${llama_size}-Clustering-l2-16

gpus=1
cpus=16
OMP_NUM_THREADS=8 srun --partition=MoE --job-name=visualize --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
  python -m smoe.entrypoint.visualization.visualize_expert_load \
  --tokenizer_path ${tokenizer_path} \
  --model_path ${model_path} \
  --data_path ${data_path} \
  --save_path ${save_path} \
  --reinit_gate ${reinit_gate} \
  --data_begin_index ${data_begin_index} \
  --data_end_index ${data_end_index} \
  --batch_size ${batch_size}
