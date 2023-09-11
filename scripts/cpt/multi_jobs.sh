models=(
    /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Clustering-l2/llama_13B-16Select4-up_proj
    /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Graph-l2_norm/llama_13B-16Select4-up_proj
    /mnt/petrelfs/share_data/quxiaoye/models/LlamaMoEForCausalLM/Random/llama_13B-16Select4-up_proj
)

for model in "${models[@]}"
do
    sbatch scripts/cpt/fpt_13b.sh $model
done


# Submitted batch job 1904066
# Submitted batch job 1904067
# Submitted batch job 1904068
