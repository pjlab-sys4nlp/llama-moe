#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base  llama_3B
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
base_model=llama_13B

num_experts=15         #  13  14  15
num_experts_residual=1 #  1  2  3
num_selects=3          #  1  2  3
expert_size=864
# 540 1080 2160 4320 8640
# 688 1376 2752 5504 11008
# 864 1728 3456 6912 13824
model_type=LlamaMoEResidualForCausalLM #  LlamaMoEResidualModel  LlamaMoEResidualForCausalLM  LlamaMoEResidualForSequenceClassification

kernel=l1_norm
criterion=max                  #  min  max
accumulate_level=sample        #  sample  total
importance_type=feature_change #  feature_grad  feature_change

tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/${base_model}/
model_path=/mnt/petrelfs/share_data/quxiaoye/models/${model_type}/Gradient-${criterion}-${kernel}-${accumulate_level}-${importance_type}/${base_model}-${num_experts}Select${num_selects}-${num_experts_residual}Residuals-${expert_size}Neurons-Share

gpus=1
cpus=8
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=test --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=spot \
  python -m smoe.entrypoint.examples.load_llama_moe_residual \
  --tokenizer_path ${tokenizer_path} \
  --model_path ${model_path} \
  --model_type ${model_type}
