#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_13B"

data_begin_index=0
data_end_index=500
batch_size=8
block_size=2048
moe_score_scale_factor=1

#save_folder=${llama_size}_dense
#model_path=/mnt/petrelfs/share_data/quxiaoye/models/llama_13B
#is_moe="False"

#save_folder=${llama_size}_moe
#model_path=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-copy/Gradient-max-l1_norm-sample-feature_change/llama_13B-16Select4-864Neurons
#is_moe="True"

#moe_score_scale_factor=5
#save_folder=${llama_size}_moe_scale${moe_score_scale_factor}
#model_path=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-copy/Gradient-max-l1_norm-sample-feature_change/llama_13B-16Select4-864Neurons
#is_moe="True"

save_folder=${llama_size}_moe_trained
model_path=/mnt/petrelfs/share_data/quxiaoye/checkpoint-18000
is_moe="True"

share_path=/mnt/petrelfs/share_data/quxiaoye
tokenizer_path=${share_path}/models/${llama_size}
data_path=${share_path}/data/vis_data/head30_shuffled_output/shuffled_20.txt
save_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/mlp-outputs-scale/${save_folder}

gpus=1
cpus=16
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=visualize --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.visualization.visualize_mlp_output_scale \
  --tokenizer_path ${tokenizer_path} \
  --model_path ${model_path} \
  --data_path ${data_path} \
  --save_path ${save_path} \
  --data_begin_index ${data_begin_index} \
  --data_end_index ${data_end_index} \
  --batch_size ${batch_size} \
  --block_size ${block_size} \
  --is_moe ${is_moe} \
  --moe_score_scale_factor ${moe_score_scale_factor}
