import os

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
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
