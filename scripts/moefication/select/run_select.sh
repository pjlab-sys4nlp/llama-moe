#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
#  ReluLLaMA-7B
llama_size="ReluLLaMA-7B"

num_experts=16            #  8  16
num_selects=4             #  2  4
split_type=Clustering-l2  #  Graph-l1_norm  Graph-l2_norm  Clustering-l2  Clustering-cos  Random
select_type=positive      #  plain  positive  l1_norm  l2_norm
mlp_init_criterion=random #  weight  random
proj_type=gate_proj       #  gate_proj  up_proj

use_balance="False"
balance_loss_lambda=0.0 # 0.0001
add_noise="False"
use_softmax="False"

data_use_percent=1.0 #  1.0  0.71  0.43
train_percent=0.97
batch_size=1024
epochs=800
lr=0.5

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
split_file_path=${data_path}/moefication_results/split/${llama_size}-${num_experts}Expert-Split-${split_type}
hidden_features_path=${data_path}/moefication_results/features/${llama_size}-Hidden-Features
save_path=${data_path}/moefication_results/select/${split_type}

save_visualization_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/expert-select/${split_type}-${select_type}-${mlp_init_criterion}/${llama_size}-${num_experts}Select${num_selects}-${proj_type}

gpus=1
cpus=16
for specify_layer in "0 1" "2 3" "4 5" "6 7" "8 9" "10 11" "12 13" "14 15" "16 17" "18 19" "20 21" "22 23" "24 25" "26 27" "28 29" "30 31"; do # 并行启用任务
  OMP_NUM_THREADS=2 srun --partition=MoE --job-name=select --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
    python -m smoe.entrypoint.moefication.llama_select_mlp \
    --model_path ${model_path} \
    --split_file_path ${split_file_path} \
    --hidden_features_path ${hidden_features_path} \
    --save_path ${save_path} \
    --save_visualization_path ${save_visualization_path} \
    --specify_layer ${specify_layer} \
    --template layers.{}.mlp.${proj_type}.weight \
    --num_experts ${num_experts} \
    --num_selects ${num_selects} \
    --select_criterion ${select_type} \
    --mlp_init_criterion ${mlp_init_criterion} \
    --use_balance ${use_balance} \
    --balance_loss_lambda ${balance_loss_lambda} \
    --add_noise ${add_noise} \
    --use_softmax ${use_softmax} \
    --data_use_percent ${data_use_percent} \
    --train_percent ${train_percent} \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --lr ${lr} & # 并行运行下一命令
  sleep 1
done
# "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31"
# "0 1" "2 3" "4 5" "6 7" "8 9" "10 11" "12 13" "14 15" "16 17" "18 19" "20 21" "22 23" "24 25" "26 27" "28 29" "30 31"
# "0 1 2 3" "4 5 6 7" "8 9 10 11" "12 13 14 15" "16 17 18 19" "20 21 22 23" "24 25 26 27" "28 29 30 31"
# "0 1 2 3 4 5 6 7" "8 9 10 11 12 13 14 15" "16 17 18 19 20 21 22 23" "24 25 26 27 28 29 30 31"
# "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" "16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
wait
