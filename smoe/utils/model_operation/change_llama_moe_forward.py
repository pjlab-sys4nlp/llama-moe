import torch
from transformers.utils import logging

from smoe.models.llama_moe import BaseMoEModelOutputWithPast
from smoe.modules.moe.moe_calculators import CalculatorOutput

logger = logging.get_logger(__name__)


def forward_universal_calculator_with_scaled_gate_score(
    self, x, topK_indices, topK_scores, expert_batch_size=None, **kwargs
) -> CalculatorOutput:
    # fmt: off
    """正向传播"""
    """临时变量"""
    batch_size = topK_indices.size(0)
    num_selects = topK_indices.size(1)
    topK_indices = topK_indices.flatten()  # shape(batch_size*num_selects)
    topK_scores = topK_scores.flatten()  # shape(batch_size*num_selects)
    batch_indices = torch.arange(batch_size, device=topK_scores.device).repeat_interleave(num_selects)  # 选出的专家编号所对应的batch编号，shape(batch_size*num_selects)

    """按照专家序号从小到大的顺序，生成专家索引"""
    _, index_sorted_topK_indices = topK_indices.sort(0)

    """按照索引重新排列scores与batch_indices，并计算专家的batch_size"""
    sorted_topK_scores = topK_scores.index_select(0, index_sorted_topK_indices)  # 各个输出对应的权重
    sorted_batch_indices = batch_indices.index_select(0, index_sorted_topK_indices)  # 各个专家对应的batch编号

    if expert_batch_size is None:
        expert_batch_size = topK_indices.bincount(minlength=self.num_experts).tolist()  # 各个专家对应的batch_size
        # if len(expert_batch_size) < self.num_experts:  # 列表长度不足专家数，说明 被选择的最大专家序号 小于 所有专家中的最大专家序号
        #     expert_batch_size.extend([0] * (self.num_experts - len(expert_batch_size)))  # 使用0补全列表

    """对每个专家重新组合batch"""
    sorted_x = x.index_select(0, sorted_batch_indices).squeeze(1)  # 将输入按照排序后的batch编号，重新编制
    split_x = torch.split(sorted_x, expert_batch_size, dim=0)  # 按照排序后每个专家的batch_size进行分隔，恰好得到各个专家所需的batch

    """各专家分别正向传播"""  # 此处应该有并行优化的空间 (如果单次forward不足以占满显卡利用率)
    expert_outputs = [self.experts(split_x[i], i) for i in range(self.num_experts) if split_x[i].shape[0] > 0]

    """重组各个专家的输出，并进行加权"""
    cat_expert_outputs = torch.cat(expert_outputs, 0)  # 拼接专家输出
    output_dim = cat_expert_outputs.size(1)
    if self.multiply_gate_scores:
        cat_expert_outputs = torch.mul(cat_expert_outputs, sorted_topK_scores.reshape(-1, 1))  # 乘权重
        #######################################
        cat_expert_outputs *= self.output_scale_factor
        #######################################

    zeros = torch.zeros((batch_size, output_dim), device=cat_expert_outputs.device, dtype=cat_expert_outputs.dtype)
    y = zeros.index_add(0, sorted_batch_indices, cat_expert_outputs)  # 按照对应的batch编号，添加输出

    return CalculatorOutput(hidden_states=y)
    # fmt: on


def forward_topk_balanced_noisy_gate_with_random_expert_selection(self, x):
    # fmt: off
    batch_size = x.shape[0]

    logits = torch.rand((batch_size, self.num_experts), device=x.device, dtype=x.dtype)
    top_logits, top_indices = logits.topk(min(self.num_selects + 1, self.num_experts), dim=1)  # 选择并排序前k+1个权重
    top_k_logits = top_logits[:, :self.num_selects]
    top_k_indices = top_indices[:, :self.num_selects]
    top_k_scores = self.softmax(top_k_logits)

    # top_k_indices = torch.stack([torch.randperm(self.num_experts, device=x.device)[:self.num_selects] for i in range(batch_size)], dim=0)
    # top_k_scores = torch.sort(self.softmax(torch.rand_like(top_k_indices, dtype=x.dtype)), dim=1, descending=True)[0]

    return {
        "topK_indices": top_k_indices,
        "topK_scores": top_k_scores,
        "balance_loss": None,
        "load": None,
        "importance": None,
    }
    # fmt: on


def forward_topk_balanced_noisy_gate_with_fixed_expert_selection(self, x):
    # fmt: off
    batch_size = x.shape[0]
    top_k_indices = torch.arange(self.num_selects, device=x.device).unsqueeze(0).repeat(batch_size, 1)
    top_k_scores = torch.sort(self.softmax(torch.rand_like(top_k_indices, dtype=x.dtype)), dim=1, descending=True)[0]

    return {
        "topK_indices": top_k_indices,
        "topK_scores": top_k_scores,
        "balance_loss": None,
        "load": None,
        "importance": None,
    }
    # fmt: on


def forward_topk_balanced_noisy_gate_with_hidden_states_recording(
    self, x, padding_mask, **kwargs
):
    # fmt: off
    self.samples_cnt += torch.sum(padding_mask).item()  ####################################

    """先计算所有专家的权重值"""
    logits = self.gate_network(x)  # gate计算出的权重

    """选出前k个权重，并计算各个专家的分数scores"""
    top_logits, top_indices = logits.topk(min(self.num_selects + 1, self.num_experts), dim=1)  # 选择并排序前k+1个权重
    top_k_logits = top_logits[:, :self.num_selects]
    top_k_indices = top_indices[:, :self.num_selects]
    top_k_scores = self.softmax(top_k_logits) if self.use_softmax else top_k_logits  # 对前k个计算softmax，得到对应的分数

    zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
    scores_filtered = zeros.scatter(dim=1, index=top_k_indices, src=top_k_scores)  # shape(batch_size, num_experts)
    scores_filtered = scores_filtered[padding_mask]  ###############################################

    """计算importance"""
    importance = scores_filtered.sum(0)  # shape(num_experts)
    self.importance_sum += scores_filtered.detach().sum(0)

    """计算load"""
    load = (scores_filtered > 0).sum(0)  # shape(num_experts)
    self.load_sum += (scores_filtered.detach() > 0).sum(0)

    """计算balance loss"""
    importance_loss = self.cv_squared(importance) * self.balance_loss_weight
    load_loss = self.cv_squared(load) * self.balance_loss_weight
    balance_loss = importance_loss + load_loss

    self.importance_loss_sum += importance_loss.detach()
    self.load_loss_sum += load_loss.detach()

    return {
        "topK_indices": top_k_indices,
        "topK_scores": top_k_scores,
        "balance_loss": balance_loss,
    }
    # fmt: on


def forward_linear_glu_moe_layer_with_padding_mask(
    self,
    x,
    padding_mask,
):
    # fmt: off
    original_shape = x.shape[:-1]
    x = x.reshape(-1, self.input_size)  # shape(batch_size*seq_len, input_size)
    padding_mask = padding_mask.reshape(-1)  # shape(batch_size*seq_len)

    gate_outputs = self.gate(x, padding_mask)  # 计算被选出的专家及其分数，以及gate的loss
    y = self.calculator(x, **gate_outputs)  # 合并各专家的计算结果

    y = y.reshape(original_shape + (self.output_size,))  # shape(batch_size, seq_len, output_size)
    return y, gate_outputs["balance_loss"]
    # fmt: on


def forward_llama_moe_decoder_with_padding_mask(
    self,
    hidden_states,
    padding_mask,  # ----- add padding_mask -----
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
):
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
    ###########################################################
    # ----- add padding_mask -----
    hidden_states, gate_loss = self.mlp(hidden_states, padding_mask)
    ###########################################################
    hidden_states = residual + hidden_states

    outputs = (
        hidden_states,
        gate_loss,
    )

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def forward_llama_moe_model_with_padding_mask(
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
    ###########################################################
    padding_mask = attention_mask.bool()  # ----- add padding_mask -----
    ###########################################################
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
    gate_loss = 0.0

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

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            ###########################################################
            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                padding_mask,  # ----- add padding_mask -----
                attention_mask,
                position_ids,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                padding_mask,  # ----- add padding_mask -----
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            ###########################################################

        hidden_states = layer_outputs[0]
        if layer_outputs[1] is not None:
            gate_loss += layer_outputs[1]

        if use_cache:
            next_decoder_cache += (layer_outputs[3 if output_attentions else 2],)

        if output_attentions:
            all_self_attns += (layer_outputs[2],)

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
        gate_loss=gate_loss,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
