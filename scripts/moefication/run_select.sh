#!/usr/bin/bash

llama_size="llama_7B" #  7B  13B  30B  base

num_experts=8
num_selects=2
select_criterion=l2_norm #  plain  positive  l2_norm
template=layers.{}.mlp.gate_proj.weight

root_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${root_path}/models/${llama_size}
split_file_path=${root_path}/moefication_results/split/${llama_size}-${num_experts}Expert-Split-Clustering
hidden_features_path=${root_path}/moefication_results/features/${llama_size}-Hidden-Features
save_path=${root_path}/moefication_results/select

gpus=1
cpus=16
for specify_layer in "0 1" "2 3" "4 5" "6 7" "8 9" "10 11" "12 13" "14 15" "16 17" "18 19" "20 21" "22 23" "24 25" "26 27" "28 29" "30 31"; do # 并行启用任务
  OMP_NUM_THREADS=8 srun --partition=MoE --job-name=select --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
    python -m smoe.entrypoint.moefication.llama_select_mlp \
    --model_path ${model_path} \
    --split_file_path ${split_file_path} \
    --hidden_features_path ${hidden_features_path} \
    --save_path ${save_path} \
    --template ${template} \
    --num_experts ${num_experts} \
    --num_selects ${num_selects} \
    --select_criterion ${select_criterion} \
    --specify_layer ${specify_layer} \
    --use_softmax & # 并行运行下一命令
done
