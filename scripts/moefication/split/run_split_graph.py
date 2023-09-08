import subprocess

# Define the bash commands


bash_commands = """
#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
llama_size=llama_13B

num_experts=16                        #  8  16
metric=l1_norm                        #  l1_norm l2_norm plain
template=layers.{}.mlp.up_proj.weight #  gate_proj  up_proj
threshold=1

data_path=/mnt/petrelfs/share_data/quxiaoye
model_path=${data_path}/models/${llama_size}
save_path=${data_path}/moefication_results/split/${llama_size}-${num_experts}Expert-Split-Graph-${metric}/
hidden_features_path=${data_path}/moefication_results/features/${llama_size}-Hidden-Features

gpus=0
cpus=16

# STEP1

for specify_layer in {0..39}; do
  OMP_NUM_THREADS=2 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
    python -m smoe.entrypoint.moefication.llama_split_graph \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --specify_layer ${specify_layer} \
    --template ${template} \
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
template2=.mlp.up_proj.weight

for layer in {0..39}; do
  OMP_NUM_THREADS=8 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
    ${gpmetis_run} ${save_path}/${template1}${layer}${template2} ${num_experts} &
  sleep 0.7
done
wait
# STEP3

template3=.part.${num_experts}

for layer in {0..39}; do
  OMP_NUM_THREADS=8 srun --partition=MoE --job-name=split --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 \
    python -m smoe.entrypoint.moefication.llama_split_graph_trans_gp \
    --gpmetised_file_path ${save_path}/${template1}${layer}${template2}${template3} &
  sleep 0.7
done
wait
chmod -R 777 ${save_path} >/dev/null 2>&1
"""

# Execute the bash commands using Python's subprocess module
subprocess.run(bash_commands, shell=True, executable="/bin/bash")
