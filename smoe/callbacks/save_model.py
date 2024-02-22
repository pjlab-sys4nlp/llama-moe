import os

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer import SCHEDULER_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from smoe.utils.vars import BEST_MODEL_CKPT_DIR, MIDDLE_MODEL_CKPT_DIR


class SaveModelCallback(TrainerCallback):
    def save_model(self, args: TrainingArguments, state: TrainerState, **kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                args.output_dir,
                PREFIX_CHECKPOINT_DIR,
                BEST_MODEL_CKPT_DIR,
                state.best_model_checkpoint,
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir,
                PREFIX_CHECKPOINT_DIR,
                MIDDLE_MODEL_CKPT_DIR,
                f"{state.global_step}",
            )

        kwargs["model"].save_pretrained(checkpoint_folder)
        kwargs["tokenizer"].save_pretrained(checkpoint_folder)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.save_model(args, state, **kwargs)
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.save_model(args, state, **kwargs)


class SavePeftModelCallback(TrainerCallback):
    def save_model(self, args, state, kwargs, peft_model_dir: str = None):
        if peft_model_dir is None:
            if state.best_model_checkpoint is not None:
                peft_model_dir = os.path.join(
                    state.best_model_checkpoint, "pt_lora_model"
                )
            else:
                peft_model_dir = os.path.join(
                    args.output_dir,
                    f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
                    "pt_lora_model",
                )

        kwargs["model"].save_pretrained(peft_model_dir)
        kwargs["tokenizer"].save_pretrained(peft_model_dir)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_dir = os.path.join(args.output_dir, "pt_lora_model")
        self.save_model(args, state, control, peft_model_dir)


class SchedulerStateCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if os.environ.get("RANK", "0") == "0":
            scheduler = kwargs.get("lr_scheduler")
            if scheduler is None:
                return
            scheduler_state = scheduler.state_dict()
            # 使用 PREFIX_CHECKPOINT_DIR 和 global_step 创建检查点目录名
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            # 完整的检查点目录路径
            checkpoint_path = os.path.join(args.output_dir, checkpoint_folder)
            # 如果目录不存在，则创建它
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            # 完整的保存路径
            save_path = os.path.join(checkpoint_path, SCHEDULER_NAME)
            # 保存scheduler状态
            torch.save(scheduler_state, save_path)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # 如果resume_from_checkpoint设置了有效路径
        if args.resume_from_checkpoint is not None:
            load_path = os.path.join(args.resume_from_checkpoint, SCHEDULER_NAME)
            # 如果该路径下有保存的调度器状态，则加载它
            if os.path.exists(load_path):
                # scheduler = kwargs['lr_scheduler']
                scheduler = kwargs.get("lr_scheduler")
                if scheduler is None:
                    return
                scheduler_state = torch.load(load_path)
                scheduler.load_state_dict(scheduler_state)
