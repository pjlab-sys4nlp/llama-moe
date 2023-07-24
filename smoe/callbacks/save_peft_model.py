import os

from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SavePeftModelCallback(TrainerCallback):
    def __init__(self, peft_model_subdir: str = "peft_model"):
        self.peft_model_subdir = peft_model_subdir

    def save_model(self, args, state, **kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, self.peft_model_subdir
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, self.peft_model_subdir)
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, **kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, self.peft_model_subdir)
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)
