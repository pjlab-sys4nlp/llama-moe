gpus=1
OMP_NUM_THREADS=16 srun --partition=MoE --job-name=test --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c 24 --job-name=convert --kill-on-bad-exit=1 \
  python smoe/entrypoint/llama_moe_example.py
