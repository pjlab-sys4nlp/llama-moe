import torch
from transformers import TrainerControl, TrainerState, TrainingArguments
from transformers.integrations import TensorBoardCallback, logger, rewrite_logs

from smoe.utils.visualization.visualize import get_heatmap_img_grid_for_tb


class EnhancedTensorboardCallback(TensorBoardCallback):
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs.update({"Total_FLOPs": state.total_flos})
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                    if "train/loss" == k:
                        tokens = state.global_step * args.num_tokens_per_batch
                        token_loss_key = "train/loss_on_tokens"
                        self.tb_writer.add_scalar(token_loss_key, v, tokens)
                elif (
                    k == "train/num_dropped_tokens"
                    and isinstance(v, tuple)
                    and all(isinstance(n, (int, float)) for n in v)
                ):
                    self.tb_writer.add_scalars(
                        f"{k}/layer",
                        {str(i): n for i, n in enumerate(v)},
                        state.global_step,
                    )
                    self.tb_writer.add_scalar(f"{k}/total", sum(v), state.global_step)
                elif (
                    k == "train/gate_load"
                    or k == "train/gate_importance"
                    and isinstance(v, tuple)
                    and all(isinstance(n, list) for n in v)
                ):
                    _v = [torch.tensor(n) for n in v]
                    self.tb_writer.add_scalars(
                        f"{k}/std/layer",
                        {str(i): n.std().item() for i, n in enumerate(_v)},
                        state.global_step,
                    )
                    self.tb_writer.add_image(
                        k, get_heatmap_img_grid_for_tb(_v), state.global_step
                    )
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()
