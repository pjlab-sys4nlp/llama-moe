llama_size="llama_7B" #  7B  13B  30B  base

num_experts=8
num_selects=2
select_type=plain # plain positive l2_norm

root_path="/mnt/petrelfs/dongdaize.d/workspace/llama-moe"
model_path=/mnt/petrelfs/share_data/quxiaoye/models/${llama_size}
split_file_path=${root_path}/llama_moe_temp_files/${llama_size}-${num_experts}Expert-Split-Clustering
select_file_path=${root_path}/llama_moe_temp_files/${llama_size}-${num_experts}Expert-Select-MLP-${select_type}
save_path=/mnt/petrelfs/share_data/quxiaoye/models/${llama_size}_MoE_${num_experts}Select${num_selects}-${select_type}

gpus=1
OMP_NUM_THREADS=16 srun --partition=MoE --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 --job-name=convert --kill-on-bad-exit=1 \
  python ${root_path}/run_moefication/llama_convert.py \
  --model_path ${model_path} \
  --split_file_path ${split_file_path} \
  --select_file_path ${select_file_path} \
  --save_path ${save_path} \
  --num_experts ${num_experts} \
  --num_selects ${num_selects}
