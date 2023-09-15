#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
base_model=llama2_7B

num_experts=16                 #  8  16
num_selects=4                  #  2  4
model_type=LlamaMoEForCausalLM #  LlamaMoEModel  LlamaMoEForCausalLM  LlamaMoEForSequenceClassification
split_type=Random              #  Graph-l1_norm  Graph-l2_norm  Clustering-l2  Clustering-cos  Random
select_type=l2_norm            #  plain  positive  l2_norm
proj_type=up_proj              #  gate_proj  up_proj

tokenizer_path=/mnt/petrelfs/share_data/quxiaoye/models/${base_model}/
model_path=/mnt/petrelfs/share_data/quxiaoye/models/${model_type}/${split_type}-${select_type}/${base_model}-${num_experts}Select${num_selects}-${proj_type}/
#model_path=/mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM-Prune/Gradient-max-l1_norm-total-feature_grad/llama2_7B-0-0.20Percent-2201Neurons

gpus=0
cpus=16
OMP_NUM_THREADS=8 srun --partition=MoE --job-name=test --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --job-name=example --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.examples.load_llama_moe \
  --tokenizer_path ${tokenizer_path} \
  --model_path ${model_path} \
  --model_type ${model_type}
