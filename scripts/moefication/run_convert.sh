#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size="llama2_7B"

num_experts=8                    #  8  16
num_selects=2                    #  2  4
convert_type=LlamaMoEForCausalLM #  LlamaMoEModel  LlamaMoEForCausalLM  LlamaMoEForSequenceClassification
split_type=Random                #  Graph  Clustering-l2  Clustering-cos  Random
select_type=l2_norm              #  plain  positive  l2_norm
proj_type=gate_proj              #  gate_proj  up_proj

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
split_file_path=${data_path}/moefication_results/split/${llama_size}-${num_experts}Expert-Split-${split_type}
select_file_path=${data_path}/moefication_results/select/${split_type}/${llama_size}-${num_experts}Expert-Select-MLP-${select_type}
save_path=${data_path}/models/${convert_type}/${split_type}-${select_type}

gpus=0
cpus=16
OMP_NUM_THREADS=8 srun --partition=MoE --job-name=convert --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
  python -m smoe.entrypoint.moefication.llama_convert \
  --model_path ${model_path} \
  --split_file_path ${split_file_path} \
  --select_file_path ${select_file_path} \
  --save_path ${save_path} \
  --template layers.{}.mlp.${proj_type}.weight \
  --num_experts ${num_experts} \
  --num_selects ${num_selects} \
  --convert_type ${convert_type}

wait
chmod -R 777 ${save_path}
