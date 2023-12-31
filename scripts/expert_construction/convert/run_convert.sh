#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
#  open_llama_7b
llama_size="llama2_7B"

num_experts=16                   #  4  8  16  32
num_selects=4                    #  1  2  4  8
convert_type=LlamaMoEForCausalLM #  LlamaMoEModel  LlamaMoEForCausalLM  LlamaMoEForSequenceClassification
split_type=Clustering-l2         #  Graph-l1_norm  Graph-l2_norm  Clustering-l2  Clustering-cos  Random
proj_type=up_proj                #  gate_proj  up_proj

score_scale_factor=4.0 #  1.0  2.0  4.0  8.0  16.0
score_scale_factor_file_path=""
#score_scale_factor_file_path=/mnt/petrelfs/dongdaize.d/workspace/train-moe/visualization/mlp-layer-wise-scale-factors/llama_13B_dense

use_default_gate=True #  True  False
select_type=l2_norm   #  plain  positive  l1_norm  l2_norm

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
split_file_path=${data_path}/moefication_results/split/${llama_size}-${num_experts}Expert-Split-${split_type}

if [ ${use_default_gate} = "True" ]; then
  select_file_path=""
  save_path=${data_path}/models/${convert_type}/${split_type}/${llama_size}-${num_experts}Select${num_selects}-${proj_type}-Scale${score_scale_factor}
else
  select_file_path=${data_path}/moefication_results/select/${split_type}/${llama_size}-${num_experts}Expert-Select-MLP-${select_type}
  save_path=${data_path}/models/${convert_type}/${split_type}-${select_type}/${llama_size}-${num_experts}Select${num_selects}-${proj_type}
fi

gpus=0
cpus=8
OMP_NUM_THREADS=2 srun --partition=MoE --job-name=convert --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=auto \
  python -m smoe.entrypoint.expert_construction.llama_convert \
  --model_path ${model_path} \
  --split_file_path ${split_file_path} \
  --select_file_path "${select_file_path}" \
  --save_path ${save_path} \
  --template layers.{}.mlp.${proj_type}.weight \
  --num_experts ${num_experts} \
  --num_selects ${num_selects} \
  --score_scale_factor ${score_scale_factor} \
  --score_scale_factor_file_path "${score_scale_factor_file_path}" \
  --convert_type ${convert_type} \
  --use_default_gate ${use_default_gate}

chmod -R 755 ${save_path} >/dev/null 2>&1
