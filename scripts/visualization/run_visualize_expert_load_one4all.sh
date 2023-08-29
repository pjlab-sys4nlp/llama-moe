#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_7B" ################ 修改修改修改修改 ################

reinit_gate=False ############## 修改修改修改修改 ##############
cpt_tokens=25B    ############ 修改修改修改修改 ############
batch_size=8

share_path=/mnt/petrelfs/share_data/quxiaoye
tokenizer_path=${share_path}/models/${llama_size}

##### Clustering-l2-16 #####
#save_folder_postfix=Clustering-l2-16
#model_path=${share_path}/models/llama_7B_MoE_16Select4-l2_norm
#model_path=${share_path}/models/tzhu_model_bak/cpt-moe-fpt-64gpus-bs16_2-zero1default-1600316/checkpoint-23000 # 100B

##### Clustering-l2-16-gate0.1 #####
save_folder_postfix=Clustering-l2-16-gate0.1
model_path=${share_path}/model_back/cpt-moe-fpt-56gpus-bs16_2-zero1default-gateloss0.1-1719794/checkpoint-7000 # 25B

##### Graph-l2-16 #####
#save_folder_postfix=Graph-l2-16
#model_path=${share_path}/models/LlamaMoEForCausalLM/Graph-l2_norm-l2_norm/llama_7B-16Select4-up_proj

##### Random-16 #####
#save_folder_postfix=Random-16
#model_path=${share_path}/models/LlamaMoEForCausalLM/Random-l2_norm/llama_7B-16Select4-up_proj

##### Initialize-16 #####
#save_folder_postfix=Initialize-16
#model_path=${share_path}/models/tzhu_model_bak/random_16select4_moe
#model_path=${share_path}/models/tzhu_model_bak/cpt-moe-fpt-7b-random-64gpus-bs16_2-zero1default-1708772-checkpoint-8000/ # 35B

save_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/expert-load-${cpt_tokens}
if [ ${reinit_gate} = "True" ]; then
  save_path=${save_path}-Ini
fi
save_path=${save_path}/${llama_size}-${save_folder_postfix}

gpus=1
cpus=16

for data_path in \
  ${share_path}/data/vis_data/wikipedia-part-003428-16de0c55-head1000.jsonl \
  ${share_path}/data/vis_data/github-part-003227-16de0c55-head1000.jsonl \
  ${share_path}/data/vis_data/commoncrawl-part-000203-16de0c55-head1000.jsonl \
  ${share_path}/data/vis_data/head30_shuffled_output/shuffled_20.txt; do
  OMP_NUM_THREADS=8 srun --partition=MoE --job-name=visualize --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
    python -m smoe.entrypoint.visualization.visualize_expert_load \
    --tokenizer_path ${tokenizer_path} \
    --model_path ${model_path} \
    --data_path ${data_path} \
    --save_path ${save_path} \
    --save_name_prefix "" \
    --reinit_gate ${reinit_gate} \
    --data_begin_index 0 \
    --data_end_index 200 \
    --batch_size ${batch_size} &
  sleep 0.7
done

for data_path in \
  ${share_path}/data/vis_data/wikipedia-part-003428-16de0c55-head1000.jsonl \
  ${share_path}/data/vis_data/github-part-003227-16de0c55-head1000.jsonl \
  ${share_path}/data/vis_data/commoncrawl-part-000203-16de0c55-head1000.jsonl; do
  OMP_NUM_THREADS=8 srun --partition=MoE --job-name=visualize --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
    python -m smoe.entrypoint.visualization.visualize_expert_load \
    --tokenizer_path ${tokenizer_path} \
    --model_path ${model_path} \
    --data_path ${data_path} \
    --save_path ${save_path} \
    --save_name_prefix "_2" \
    --reinit_gate ${reinit_gate} \
    --data_begin_index 200 \
    --data_end_index 400 \
    --batch_size ${batch_size} &
  sleep 0.7
done
