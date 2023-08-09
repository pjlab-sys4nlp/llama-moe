#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama2_7B"

gpus=0
cpus=4

# 可视化所有可能的结果组合，无效进程会自动报错退出
for num_experts in 8 16; do
  for split_type in "Clustering-l2" "Clustering-cos" "Random"; do
    for select_type in "plain" "positive" "l2_norm"; do

      result_path=/mnt/petrelfs/share_data/quxiaoye/moefication_results/select/${split_type}/${llama_size}-${num_experts}Expert-Select-MLP-${select_type}
      save_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization-scheduler-train/${split_type}/${llama_size}-${num_experts}Expert-Select-MLP-${select_type}

      OMP_NUM_THREADS=8 srun --partition=MoE --job-name=visualize --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
        python -m smoe.entrypoint.visualization.visualize_expert_select_mlp \
        --result_path ${result_path} \
        --save_path ${save_path} & # 并行运行下一命令
      sleep 0.5 # 等待0.5s

    done
  done
done
