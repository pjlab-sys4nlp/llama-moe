#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_7B"

proj_type=up_proj         #  gate_proj  up_proj
visualize_criterion=l2_norm #  plain  l1_norm  l2_norm

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
hidden_features_path=${data_path}/moefication_results/features/${llama_size}-Hidden-Features

save_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/swiglu-output/${llama_size}/${proj_type}-${visualize_criterion}

gpus=1
cpus=16
for specify_layer in "0 1 2 3" "4 5 6 7" "8 9 10 11" "12 13 14 15" "16 17 18 19" "20 21 22 23" "24 25 26 27" "28 29 30 31"; do # 并行启用任务
  OMP_NUM_THREADS=8 srun --partition=MoE --job-name=visualize --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
    python -m smoe.entrypoint.visualization.visualize_swiglu_output \
    --model_path ${model_path} \
    --hidden_features_path ${hidden_features_path} \
    --save_path ${save_path} \
    --template layers.{}.mlp.${proj_type}.weight \
    --specify_layer ${specify_layer} \
    --visualize_criterion ${visualize_criterion} & # 并行运行下一命令
  sleep 0.5                                        # 等待0.5s
done
# "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31"
# "0 1" "2 3" "4 5" "6 7" "8 9" "10 11" "12 13" "14 15" "16 17" "18 19" "20 21" "22 23" "24 25" "26 27" "28 29" "30 31"
# "0 1 2 3" "4 5 6 7" "8 9 10 11" "12 13 14 15" "16 17 18 19" "20 21 22 23" "24 25 26 27" "28 29 30 31"
# "0 1 2 3 4 5 6 7" "8 9 10 11 12 13 14 15" "16 17 18 19 20 21 22 23" "24 25 26 27 28 29 30 31"
# "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" "16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
