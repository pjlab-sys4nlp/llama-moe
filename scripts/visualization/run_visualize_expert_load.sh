#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_7B"

batch_size=8

share_path=/mnt/petrelfs/share_data/quxiaoye
tokenizer_path=${share_path}/models/${llama_size}
model_path=${share_path}/models/tzhu_model_bak/cpt-moe-fpt-64gpus-bs16_2-zero1default-1600316/checkpoint-23000
save_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/expert-load/${llama_size}

#  commoncrawl-part-000203-16de0c55-head1000.jsonl
#  github-part-003227-16de0c55-head1000.jsonl
#  wikipedia-part-003428-16de0c55-head1000.jsonl
data_name=wikipedia-part-003428-16de0c55-head1000.jsonl
data_path=${share_path}/data/vis_data/${data_name}
#data_path=/mnt/petrelfs/share_data/quxiaoye/part-003190-16de0c55.jsonl


gpus=1
cpus=16
OMP_NUM_THREADS=8 srun --partition=MoE --job-name=visualize --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
  python -m smoe.entrypoint.visualization.visualize_expert_load \
  --tokenizer_path ${tokenizer_path} \
  --model_path ${model_path} \
  --data_path ${data_path} \
  --save_path ${save_path} \
  --batch_size ${batch_size}
