#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
#  open_llama_7b
llama_size="open_llama_7b"

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
save_path=${data_path}/moefication_results/split

gpus=0
cpus=8
for num_experts in 4 8 16 32; do
  for proj_type in "gate_proj" "up_proj"; do
    OMP_NUM_THREADS=2 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
      python -m smoe.entrypoint.moefication.llama_split_random \
      --model_path ${model_path} \
      --save_path ${save_path} \
      --template layers.{}.mlp.${proj_type}.weight \
      --num_experts ${num_experts} &
    sleep 0.7
  done
done

wait
chmod -R 755 ${save_path} >/dev/null 2>&1
