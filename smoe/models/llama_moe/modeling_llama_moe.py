""" PyTorch LLaMA-MoE model."""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.utils import ModelOutput, logging

from smoe.models.llama_moe.configuration_llama_moe import LlamaMoEConfig
from smoe.modules.moe.moe_layers import LinearGLUMoELayer, MoEMlpOutput

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaMoEConfig"


@dataclass
class MoEDecoderLayerOutput(ModelOutput):
    # zhutong: do not change the order of these fields!!
    hidden_states: Optional[torch.FloatTensor] = None
    balance_loss: Optional[float] = None
    num_dropped_tokens: Optional[torch.Tensor] = None
    gate_load: Optional[list[torch.Tensor]] = None
    gate_importance: Optional[list[torch.Tensor]] = None
    self_attn_weights: Optional[torch.FloatTensor] = None
    present_key_value: Optional[torch.FloatTensor] = None


@dataclass
class BaseMoEModelOutputWithPast(ModelOutput):
    """
    Args:
        num_dropped_tokens: layer idx to the number of dropped tokens
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    balance_loss: Optional[float] = None
    num_dropped_tokens: Optional[Tuple[torch.Tensor]] = None
    gate_load: Optional[Tuple[list]] = None
    gate_importance: Optional[Tuple[list]] = None


@dataclass
class MoECausalLMOutputWithPast(CausalLMOutputWithPast):
    balance_loss: Optional[float] = None
    num_dropped_tokens: Optional[Tuple[int]] = None
    gate_load: Optional[Tuple[list]] = None
    gate_importance: Optional[Tuple[list]] = None


class LlamaMoEDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaMoEConfig, layer_index):
        super().__init__(config)

        gating_config = {
            # all gates
            "gate_type": config.gate_type,
            "gate_network": config.gate_network,
            "gate_use_softmax": config.gate_use_softmax,
            "gate_use_balance": config.gate_use_balance,
            "gate_balance_loss_weight": config.gate_balance_loss_weight,
            "gate_add_noise": config.gate_add_noise,
            # TopKBalancedNoisyGate
            "gate_noise_epsilon": config.gate_noise_epsilon,
        }
        calculator_config = {
            # all calculators
            "calculator_type": config.calculator_type,
            "multiply_gate_scores": config.multiply_gate_scores,
            "score_scale_factor": config.score_scale_factor[layer_index]
            if isinstance(config.score_scale_factor, list)
            else config.score_scale_factor,
            # SwitchDropTokenCalculator
            "drop_tokens": config.drop_tokens,
            "dropped_padding": config.dropped_padding,
            "capacity_factor": config.capacity_factor,
        }

        self.mlp = LinearGLUMoELayer(
            input_size=self.hidden_size,
            hidden_size=config.intermediate_size,
            output_size=self.hidden_size,
            hidden_act=config.hidden_act,
            num_experts=config.num_experts,
            num_selects=config.num_selects,
            size_experts=(
                config.size_experts[layer_index]
                if config.size_experts is not None
                else None
            ),
            bias=False,
            **gating_config,
            **calculator_config,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ) -> MoEDecoderLayerOutput:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_outs: MoEMlpOutput = self.mlp(hidden_states)
        hidden_states = residual + mlp_outs.hidden_states

        outputs = (
            hidden_states,
            mlp_outs.balance_loss,
            mlp_outs.num_dropped_tokens,
            mlp_outs.gate_load,
            mlp_outs.gate_importance,
        )
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        for i, _o in enumerate(outputs):
            if not isinstance(_o, torch.Tensor):
                raise RuntimeError(
                    f"outputs[{i}]({type(_o)}) should be torch.Tensor to support grad ckpt"
                )

        return outputs

    def set_moe_num_selects(self, num_selects):
        self.mlp.set_num_selects(num_selects)

    def set_moe_gate_use_softmax(self, use_softmax):
        self.mlp.set_gate_use_softmax(use_softmax)

    def set_moe_gate_use_balance(self, use_balance):
        self.mlp.set_gate_use_balance(use_balance)

    def set_moe_gate_balance_loss_weight(self, balance_loss_weight):
        self.mlp.set_gate_balance_loss_weight(balance_loss_weight)

    def set_moe_gate_add_noise(self, add_noise):
        self.mlp.set_gate_add_noise(add_noise)

    def set_moe_gate_noise_epsilon(self, noise_epsilon):
        self.mlp.set_gate_noise_epsilon(noise_epsilon)

    def set_moe_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.mlp.set_calculator_multiply_gate_scores(multiply_gate_scores)

    def set_moe_calculator_score_scale_factor(self, score_scale_factor):
        self.mlp.set_calculator_score_scale_factor(score_scale_factor)

    def set_moe_calculator_drop_tokens(self, drop_tokens):
        self.mlp.set_calculator_drop_tokens(drop_tokens)

    def set_moe_calculator_dropped_padding(self, dropped_padding):
        self.mlp.set_calculator_dropped_padding(dropped_padding)

    def set_moe_calculator_capacity_factor(self, capacity_factor):
        self.mlp.set_calculator_capacity_factor(capacity_factor)

    def reset_gate_network(self):
        self.mlp.reset_gate_network()


class LlamaMoEPreTrainedModel(LlamaPreTrainedModel):
    config_class = LlamaMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True  # added
    _no_split_modules = ["LlamaMoEDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaMoEModel):
            module.gradient_checkpointing = value


class LlamaMoEModel(LlamaModel, LlamaMoEPreTrainedModel):
    def __init__(self, config: LlamaMoEConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaMoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at"
                " the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds
        balance_loss = 0.0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing."
                    " Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        num_dropped_tokens = ()
        gate_load = ()
        gate_importance = ()
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs: tuple = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs: tuple = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            layer_outputs = MoEDecoderLayerOutput(*layer_outputs)

            hidden_states = layer_outputs.hidden_states
            if layer_outputs.balance_loss is not None:
                balance_loss += layer_outputs.balance_loss

            if use_cache:
                next_decoder_cache += (layer_outputs.present_key_value,)

            if output_attentions:
                all_self_attns += (layer_outputs.self_attn_weights,)

            num_dropped_tokens += (layer_outputs.num_dropped_tokens,)
            gate_load += (layer_outputs.gate_load,)
            gate_importance += (layer_outputs.gate_importance,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseMoEModelOutputWithPast(
            last_hidden_state=hidden_states,
            balance_loss=balance_loss,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            num_dropped_tokens=num_dropped_tokens,
            gate_load=gate_load,
            gate_importance=gate_importance,
        )

    def set_moe_num_selects(self, num_selects):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_num_selects(num_selects)

    def set_moe_gate_use_softmax(self, use_softmax):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_use_softmax(use_softmax)

    def set_moe_gate_use_balance(self, use_balance):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_use_balance(use_balance)

    def set_moe_gate_balance_loss_weight(self, balance_loss_weight):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_balance_loss_weight(balance_loss_weight)

    def set_moe_gate_add_noise(self, add_noise):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_add_noise(add_noise)

    def set_moe_gate_noise_epsilon(self, noise_epsilon):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_gate_noise_epsilon(noise_epsilon)

    def set_moe_calculator_multiply_gate_scores(self, multiply_gate_scores):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_calculator_multiply_gate_scores(multiply_gate_scores)

    def set_moe_calculator_score_scale_factor(
        self, score_scale_factor, layer_index=None
    ):
        if layer_index is None:
            for idx, decoder_layer in enumerate(self.layers):
                decoder_layer.set_moe_calculator_score_scale_factor(score_scale_factor)
        else:
            self.layers[layer_index].set_moe_calculator_score_scale_factor(
                score_scale_factor
            )

    def set_moe_calculator_drop_tokens(self, drop_tokens):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_calculator_drop_tokens(drop_tokens)

    def set_moe_calculator_dropped_padding(self, dropped_padding):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_calculator_dropped_padding(dropped_padding)

    def set_moe_calculator_capacity_factor(self, capacity_factor):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.set_moe_calculator_capacity_factor(capacity_factor)

    def reset_gate_network(self):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.reset_gate_network()


class LlamaMoEForCausalLM(LlamaForCausalLM, LlamaMoEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaMoEModel(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseMoEModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if outputs.balance_loss is not None:
                loss += outputs.balance_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            num_dropped_tokens=outputs.num_dropped_tokens,
            balance_loss=outputs.balance_loss,
            gate_load=outputs.gate_load,
            gate_importance=outputs.gate_importance,
        )

    def set_moe_num_selects(self, num_selects):
        self.model.set_moe_num_selects(num_selects)

    def set_moe_gate_use_softmax(self, use_softmax):
        self.model.set_moe_gate_use_softmax(use_softmax)

    def set_moe_gate_use_balance(self, use_balance):
        self.model.set_moe_gate_use_balance(use_balance)

    def set_moe_gate_balance_loss_weight(self, balance_loss_weight):
        self.model.set_moe_gate_balance_loss_weight(balance_loss_weight)

    def set_moe_gate_add_noise(self, add_noise):
        self.model.set_moe_gate_add_noise(add_noise)

    def set_moe_gate_noise_epsilon(self, noise_epsilon):
        self.model.set_moe_gate_noise_epsilon(noise_epsilon)

    def set_moe_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.model.set_moe_calculator_multiply_gate_scores(multiply_gate_scores)

    def set_moe_calculator_score_scale_factor(
        self, score_scale_factor, layer_index=None
    ):
        self.model.set_moe_calculator_score_scale_factor(
            score_scale_factor, layer_index=layer_index
        )

    def set_moe_calculator_drop_tokens(self, drop_tokens):
        self.model.set_moe_calculator_drop_tokens(drop_tokens)

    def set_moe_calculator_dropped_padding(self, dropped_padding):
        self.model.set_moe_calculator_dropped_padding(dropped_padding)

    def set_moe_calculator_capacity_factor(self, capacity_factor):
        self.model.set_moe_calculator_capacity_factor(capacity_factor)

    def reset_gate_network(self):
        self.model.reset_gate_network()


class LlamaMoEForSequenceClassification(
    LlamaForSequenceClassification, LlamaMoEPreTrainedModel
):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaMoEModel(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        balance_loss = transformer_outputs[1]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if loss is not None and balance_loss is not None:
            loss += balance_loss
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def set_moe_num_selects(self, num_selects):
        self.model.set_moe_num_selects(num_selects)

    def set_moe_gate_use_softmax(self, use_softmax):
        self.model.set_moe_gate_use_softmax(use_softmax)

    def set_moe_gate_use_balance(self, use_balance):
        self.model.set_moe_gate_use_balance(use_balance)

    def set_moe_gate_balance_loss_weight(self, balance_loss_weight):
        self.model.set_moe_gate_balance_loss_weight(balance_loss_weight)

    def set_moe_gate_add_noise(self, add_noise):
        self.model.set_moe_gate_add_noise(add_noise)

    def set_moe_gate_noise_epsilon(self, noise_epsilon):
        self.model.set_moe_gate_noise_epsilon(noise_epsilon)

    def set_moe_calculator_multiply_gate_scores(self, multiply_gate_scores):
        self.model.set_moe_calculator_multiply_gate_scores(multiply_gate_scores)

    def set_moe_calculator_score_scale_factor(
        self, score_scale_factor, layer_index=None
    ):
        self.model.set_moe_calculator_score_scale_factor(
            score_scale_factor, layer_index=layer_index
        )

    def set_moe_calculator_drop_tokens(self, drop_tokens):
        self.model.set_moe_calculator_drop_tokens(drop_tokens)

    def set_moe_calculator_dropped_padding(self, dropped_padding):
        self.model.set_moe_calculator_dropped_padding(dropped_padding)

    def set_moe_calculator_capacity_factor(self, capacity_factor):
        self.model.set_moe_calculator_capacity_factor(capacity_factor)

    def reset_gate_network(self):
        self.model.reset_gate_network()
