llama_size="llama_7B" #  7B  13B  30B  base

num_experts=8

root_path="/mnt/petrelfs/dongdaize.d/workspace/llama-moe"
model_path=/mnt/petrelfs/share_data/quxiaoye/models/${llama_size}
save_path=${root_path}/llama_moe_temp_files

gpus=1
OMP_NUM_THREADS=64 srun --partition=MoE --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 --job-name=split --kill-on-bad-exit=1 \
  python ${root_path}/run_moefication/llama_split_clustering.py \
  --model_path ${model_path} \
  --save_path ${save_path} \
  --num_experts ${num_experts}
