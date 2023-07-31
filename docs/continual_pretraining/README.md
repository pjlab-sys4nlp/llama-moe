# 🚅 Training Guide

## ⚙️ Configuration Instructions

For `scripts/cpt/lora.sh` and `scripts/cpt/fpt.sh` files, we could run an experiment via `sbatch`. e.g. `sbatch scripts/cpt/lora.sh` .

The `sbatch` command is similar to `nohup`, and the submitted job would be running on the background.

Here are some instructions for slurm configuration items:
- `--job-name`: the name of a job, could be changed at your convenience.
- `--partition`: the slurm partition. For MoE group, the default partition is `MoE` .
- `--output`: the logging file of `stdout`. `%x` means the job name, and `%j` is the job ID.
- `--error`: the logging file of `stderr`.
- `--ntasks-per-node`: always set to `1` .
- `--cpus-per-task`: may change according to different usage. The maximum CPUs for a node is `128` (according to `cinfo -p MoE`).
- `--nodes`: how many nodes would you like to use. **NOTICE:** the value of `num_nodes` must be the same with `--nodes` .
- `--gres`: the number of GPUs for each node (in a format of `gpu:<num>`, e.g. `gpu:8`). **NOTICE:** the value of `num_gpu_per_node` must agree with `--gres`.

## 📦 Training Different Models

`model_type` and `pretrained_model` in `scripts/cpt/(lora|fpt).sh` help specify the fundation model.

For vanilla LLaMA, use the following settings:
```bash
model_type="llama"
pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B
```

For LLaMA with MoEfication, use the following settings:
```bash
model_type="llama_moe"
pretrained_model=/mnt/petrelfs/share_data/quxiaoye/models/llama_7B_MoE_16Select4-l2_norm
```

## 🧮 Estimation of Training Speed and Tokens

For convenient estimation of the model training speed, we provide some useful information at the very beginning of log files:

```log
max_steps: 63578
global batch size: 768
#tokens/batch: 1572864
```

- `global batch size`: per_device_train_batch_size * gradient_accumulation_steps * num_nodes * num_gpu_per_node
- `max_steps`: how many steps should be trained to reach 100B tokens: 10^11 / (block_size * per_device_train_batch_size * gradient_accumulation_steps * num_nodes * num_gpu_per_node)
- `#tokens/batch`: the number of trained tokens for one global batch

When estimating the expected time, you may want to check the `running time/step` via tensorboard or from the logging file.

Based on the above information, the expected time could be calculated.

## 🛹 Tracking Loss with Tensorboard

The tensorboard `logging_dir` could be found at `outputs/<job-name>-<job-id>/runs/<logging-dir>`.

For example, if my job name is `cpt-moe-fpt-bs16-48gpus` in the sbatch file, the tensorboard could be started from that by: `tensorboard --logdir outputs/cpt-moe-fpt-bs16-48gpus-1535835/runs/Jul31_14-12-00_SH-IDCA1404-10-140-54-100` .

For multiple tasks with different logging directories, you could run the following command:

```bash
$ tensorboard --logdir_spec short_name:dir1,short_name2:dir2 --port 8001
```

Here, the `short_name` is an abbreviation for your task, and the port number could be changed manually if there's a port conflict. e.g.

```bash
$ tensorboard --logdir_spec moe_from_scratch:outputs/cpt-llama-moe-scratch-lora-bs16-1476932/runs/Jul26_21-53-42_SH-IDCA1404-10-140-54-121,moe_lora:outputs/cpt-llama-lora-bs16-1476918/runs/Jul26_21-31-09_SH-IDCA1404-10-140-54-122 --port 8001
```
