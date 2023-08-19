from transformers.integrations import TensorBoardCallback, logger, rewrite_logs


class EnhancedTensorboardCallback(TensorBoardCallback):
    def on_log(self, args, state, control, logs: dict = None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs.update({"Total_FLOPs": state.get("total_flos", -1)})
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                    if "train/loss" == k:
                        tokens = state.global_step * args.num_tokens_per_batch
                        token_loss_key = "train/loss_on_tokens"
                        self.tb_writer.add_scalar(token_loss_key, v, tokens)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()
