#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama_7B"

kernel=l1_norm
proj_type=up_proj

expert_index=0
use_grad_sum=True

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
save_path=${data_path}/moefication_results/prune

gpus=0
cpus=8
for retain_percent in 0.99 0.98 0.95 0.90 0.80 0.60 0.40 0.25 0.20; do
  for criterion in min max; do
    for accumulate_level in sample total; do
      for importance_type in feature_grad feature_change; do
        grad_file_path=${data_path}/moefication_results/split/Gradients/${llama_size}-Gradients-${kernel}-${accumulate_level}-${importance_type}

        OMP_NUM_THREADS=8 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
          python -m smoe.entrypoint.moefication.llama_prune_gradient \
          --model_path ${model_path} \
          --grad_file_path ${grad_file_path} \
          --save_path ${save_path} \
          --retain_percent ${retain_percent} \
          --expert_index ${expert_index} \
          --template layers.{}.mlp.${proj_type}.weight \
          --kernel ${kernel} \
          --accumulate_level ${accumulate_level} \
          --importance_type ${importance_type} \
          --criterion ${criterion} \
          --use_grad_sum ${use_grad_sum} &
        sleep 1

      done
    done
  done
done

wait
chmod -R 777 ${save_path} >/dev/null 2>&1
