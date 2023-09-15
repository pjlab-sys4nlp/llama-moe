#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_7B"

# 定义 num_selects 数组，与 num_experts 一一对应
declare -a num_experts_array=(8 16)
declare -a num_selects_array=(2 4)

# 可视化所有可能的结果组合，无效进程会自动报错退出
gpus=0
cpus=4
for idx in "${!num_selects_array[@]}"; do
  num_experts="${num_experts_array[$idx]}"
  num_selects="${num_selects_array[$idx]}"
  for split_type in "Graph-l1_norm" "Graph-l2_norm" "Clustering-l2" "Clustering-cos" "Random"; do
    for select_type in "plain" "positive" "l1_norm" "l2_norm"; do
      for proj_type in "gate_proj" "up_proj"; do

        result_path=/mnt/petrelfs/share_data/quxiaoye/moefication_results/select/${split_type}/${llama_size}-${num_experts}Expert-Select-MLP-${select_type}
        save_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/expert-select/${split_type}-${select_type}/${llama_size}-${num_experts}Select${num_selects}-${proj_type}

        # 若result_path存在，则执行可视化
        if [ -d "$result_path" ]; then
          OMP_NUM_THREADS=2 srun --partition=MoE --job-name=visualize --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
            python -m smoe.entrypoint.visualization.visualize_expert_select_mlp \
            --result_path ${result_path} \
            --save_path ${save_path} \
            --proj_type ${proj_type} & # 并行运行下一命令
          sleep 0.5                    # 等待0.5s
        else
          echo "Directory does not exist: $result_path"
        fi

      done
    done
  done
done
