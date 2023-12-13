#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size=llama_13B

num_experts=16                        #  8  16
metric=l1_norm                        #  l1_norm l2_norm plain
proj_type=up_proj #  gate_proj  up_proj
threshold=1

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
save_path=${data_path}/moefication_results/split/${llama_size}-${num_experts}Expert-Split-Graph-${metric}/
hidden_features_path=${data_path}/moefication_results/features/${llama_size}-Hidden-Features

gpus=0
cpus=16

# STEP1

for specify_layer in {0..39}; do
  OMP_NUM_THREADS=2 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
    python -m smoe.entrypoint.moefication.llama_split_graph \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --specify_layer ${specify_layer} \
    --template layers.{}.mlp.${proj_type}.weight \
    --num_experts ${num_experts} \
    --threshold ${threshold} \
    --metric ${metric} \
    --hidden_features_path ${hidden_features_path} &
  sleep 0.7
done
wait

# STEP2

gpmetis_run=/mnt/petrelfs/share_data/quxiaoye/metis_for_graph_split/bin/gpmetis
template1=layers.
template2=.mlp.${proj_type}.weight

for layer in {0..39}; do
  OMP_NUM_THREADS=8 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
    ${gpmetis_run} ${save_path}/${template1}${layer}${template2} ${num_experts} &
  sleep 0.7
done
wait

# STEP3

template3=.part.${num_experts}

for layer in {0..39}; do
  OMP_NUM_THREADS=8 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
    python -m smoe.entrypoint.moefication.llama_split_graph_trans_gp \
    --gpmetised_file_path ${save_path}/${template1}${layer}${template2}${template3} &
  sleep 0.7
done
wait

chmod -R 755 ${save_path} >/dev/null 2>&1
