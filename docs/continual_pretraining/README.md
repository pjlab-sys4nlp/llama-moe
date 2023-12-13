# 🚅 Training Guide

## 🗞️ Executive Scripts

| Description               | Path                                                                                   |
| :------------------------ | :------------------------------------------------------------------------------------- |
| LLaMA-MoE 2/16 Experts    | `scripts/cpt/16_2/baseline_112gpus_sheared_llama_portion_fluency_sf8.sh`               |
| LLaMA-MoE 4/16 Experts    | `scripts/cpt/dynamic_data_selection/baseline_112gpus_sheared_llama_portion_fluency.sh` |
| Dynamic<sub>Sheared</sub> | `scripts/cpt/dynamic_data_selection/sheared_llama_112gpus.sh`                          |

## 🌴 Other Arguments in Executive Scripts

| Argument Name                         | Description                                                                                                                                        |
| :------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dynamic_data_selection`            | For different dynamic data sampling strategies, choose one from: `sheared_llama` or `none` (static). Default: `none`                               |
| `--moe_calculator_score_scale_factor` | Scale factor to multiply after hidden states are procesed by experts. Should be $\frac{\text{\#total experts}}{\text{\#selected}}$. Default: `4.0` |
| `--num_selects`                       | The number of selected experts. Default: `4`                                                                                                       |
| `--gate_balance_loss_weight`          | The weight of the balance loss for the gate. Default: `1e-2`                                                                                       |

## 📋 Checklist before Starting an Experiment

- [ ] balance loss weight
- [ ] scale factor
- [ ] learning rate
- [ ] warmup steps
- [ ] evaluation steps
- [ ] logging steps
- [ ] global batch size
- [ ] number of selected experts
- [ ] pretrained model
- [ ] data path
- [ ] GPUs
- [ ] comment

## ⚙️ Configuration Instructions for Slurm Users

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

llama1-7b 16 select 4: 3.49b params

llama1-13b total params: 13,015,864,320 - total mlp params:  8,493,465,600

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
