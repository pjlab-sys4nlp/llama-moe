import math
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    learning_rate: float,
    final_lr_portion: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        learning_rate * final_lr_portion,
        0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
    )


class LlamaLrSchedulingTrainer(Trainer):
    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
        lr_lambda = partial(
            _get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
            learning_rate=self.args.learning_rate,
            final_lr_portion=self.args.final_lr_portion,
        )
        last_epoch = -1
        self.lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch)
        return self.lr_scheduler
