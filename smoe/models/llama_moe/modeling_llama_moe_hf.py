import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.distributions.normal import Normal
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)

from .configuration_llama_moe import LlamaMoEConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaMoEConfig"


@dataclass
class CalculatorOutput(ModelOutput):
    hidden_states: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[int] = None


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
    balance_loss: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[Tuple[torch.Tensor]] = None
    gate_load: Optional[torch.LongTensor] = None
    gate_importance: Optional[torch.FloatTensor] = None
    expert2tokens: Optional[dict] = None


@dataclass
class MoECausalLMOutputWithPast(CausalLMOutputWithPast):
    balance_loss: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[Tuple[int]] = None
    gate_load: Optional[Tuple[torch.LongTensor]] = None
    gate_importance: Optional[Tuple[torch.FloatTensor]] = None
    expert2tokens: Optional[dict] = None


@dataclass
class MoEMlpOutput(ModelOutput):
    hidden_states: Optional[torch.FloatTensor] = None
    balance_loss: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[int] = None
    gate_load: Optional[torch.LongTensor] = None
    gate_importance: Optional[torch.FloatTensor] = None
    expert2tokens: Optional[dict] = None


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaMoEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class TopKBalancedNoisyGate(nn.Module):
    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        gate_network="mlp",
        use_softmax=True,
        use_balance=True,
        balance_loss_weight=1e-2,
        add_noise=True,
        noise_epsilon=1e-2,
    ):
        super(TopKBalancedNoisyGate, self).__init__()
        assert num_selects <= num_experts
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_network_type = gate_network
        self.gate_network = self.get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(1)

        self.use_balance = use_balance
        self.balance_loss_weight = balance_loss_weight

        # add_noise
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.warned = False
        if self.add_noise:
            self.weight_noise = nn.Linear(input_size, num_experts, bias=False)
            self.weight_noise.weight.data = torch.zeros(
                (num_experts, input_size),
                requires_grad=True,
                device=self.weight_noise.weight.data.device,
                dtype=self.weight_noise.weight.data.dtype,
            )
            self.mean = 0.0
            self.std = 1.0
            self.normal = Normal(self.mean, self.std)
            self.softplus = nn.Softplus()

        self.reset_parameters()

    def get_gate_network(self, gate_type, input_size, num_experts):
        gate_type = gate_type.lower()

        if gate_type == "linear":
            gate_network = nn.Linear(input_size, num_experts, bias=False)
            nn.init.zeros_(gate_network.weight)
        elif gate_type == "mlp":
            gate_network = torch.nn.Sequential(
                torch.nn.Linear(input_size, num_experts, bias=False),
                torch.nn.Tanh(),
                torch.nn.Linear(num_experts, num_experts, bias=False),
            )
        else:
            raise ValueError(f"Unexpected gate_type: {gate_type}.")

        return gate_network

    def reset_gate_network(self):
        if "gate_network_type" not in vars(self):
            raise KeyError(f"{type(self)} does not have a gate network.")
        else:
            self.gate_network = self.get_gate_network(
                self.gate_network_type, self.input_size, self.num_experts
            )

    def reset_parameters(self):
        if self.add_noise:
            nn.init.zeros_(self.weight_noise.weight)
            # nn.init.zeros_(self.weight_noise)

    def cv_squared(self, x, eps=1e-10):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.s
        """
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x):
        logits_gate = self.gate_network(x)
        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)
            noise_control = self.softplus(noise_mm) + self.noise_epsilon
            logits_noise = torch.randn_like(logits_gate) * noise_control
            logits = logits_gate + logits_noise
        else:
            logits = logits_gate

        top_logits, top_indices = logits.topk(
            min(self.num_selects + 1, self.num_experts), dim=1
        )  # 选择并排序前k+1个权重
        top_k_logits = top_logits[:, : self.num_selects]
        top_k_indices = top_indices[:, : self.num_selects]
        top_k_scores = (
            self.softmax(top_k_logits.to(torch.float32))
            if self.use_softmax
            else top_k_logits
        )
        top_k_scores = top_k_scores.to(logits.dtype)

        zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
        scores_filtered = zeros.scatter(
            dim=1, index=top_k_indices, src=top_k_scores
        )  # shape(batch_size, num_experts)
        importance = scores_filtered.sum(0)  # shape(num_experts)

        if self.training:
            if self.add_noise and self.num_selects != self.num_experts:
                batch_size = top_logits.size(0)
                m = top_logits.size(1)
                top_values_flat = top_logits.flatten()
                threshold_positions_if_in = (
                    torch.arange(batch_size, device=x.device) * m + self.num_selects
                )
                threshold_if_in = torch.unsqueeze(
                    torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
                )
                is_in = torch.gt(logits_noise, threshold_if_in)
                threshold_positions_if_out = threshold_positions_if_in - 1
                threshold_if_out = torch.unsqueeze(
                    torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
                )
                # is each value currently in the top k.
                prob_if_in = self.normal.cdf(
                    (logits_gate - threshold_if_in) / noise_control
                )
                prob_if_out = self.normal.cdf(
                    (logits_gate - threshold_if_out) / noise_control
                )
                prob = torch.where(is_in, prob_if_in, prob_if_out)
                load = prob.sum(0)
            else:
                load = (scores_filtered > 0).sum(0)
                if not self.add_noise and not self.warned:
                    warnings.warn(
                        'Gradient-trackable implementation for load calculation is only available when "add_noise=True". '
                        'Training without noise will block the gradient from "load" path and lead to inconsistency in optimization objectives.'
                    )
                    self.warned = True
        else:
            load = (scores_filtered > 0).sum(0)

        if self.use_balance:
            balance_loss = self.cv_squared(importance) + self.cv_squared(load)
            balance_loss *= self.balance_loss_weight
        else:
            balance_loss = torch.tensor(-100.0, device=x.device)

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": balance_loss,
            "load": load,
            "importance": importance,
        }


class LinearGLUExperts(nn.Module):
    """
    Modified from transformers.models.llama.modeling_llama.LlamaMLP
    """

    __constants__ = [
        "bias",
        "in_features",
        "hidden_features",
        "out_features",
        "hidden_act",
        "num_experts",
        "size_experts",
    ]

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_act,
        num_experts,
        size_experts=None,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LinearGLUExperts, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.hidden_act = hidden_act
        self.num_experts = num_experts

        if size_experts is None:
            # all experts share the same number of hidden neurons
            assert hidden_features % num_experts == 0
            size_per_expert = hidden_features // num_experts
            size_experts = [size_per_expert for _ in range(num_experts)]
        else:
            # use specified expert sizes
            assert (
                len(size_experts) == num_experts
                and sum(size_experts) == hidden_features
            )
        self.size_experts = size_experts

        self.act_fn = ACT2FN[hidden_act]

        self.weight_gate = nn.ParameterList()
        self.weight_up = nn.ParameterList()
        self.weight_down = nn.ParameterList()

        for i in range(num_experts):
            # this matrix will be transposed when performing linear forwarding
            this_expert_weight_gate = nn.Parameter(
                torch.empty((size_experts[i], in_features), **factory_kwargs)
            )
            # this matrix will be transposed when performing linear forwarding
            this_expert_weight_up = nn.Parameter(
                torch.empty((size_experts[i], in_features), **factory_kwargs)
            )
            # this matrix will be transposed when performing linear forwarding
            this_expert_weight_down = nn.Parameter(
                torch.empty((out_features, size_experts[i]), **factory_kwargs)
            )
            self.weight_gate.append(this_expert_weight_gate)
            self.weight_up.append(this_expert_weight_up)
            self.weight_down.append(this_expert_weight_down)

        if bias:
            self.bias_gate = nn.ParameterList()
            self.bias_up = nn.ParameterList()
            self.bias_down = nn.ParameterList()

            for i in range(num_experts):
                this_expert_bias_gate = nn.Parameter(
                    torch.empty((size_experts[i],), **factory_kwargs)
                )
                this_expert_bias_up = nn.Parameter(
                    torch.empty((size_experts[i],), **factory_kwargs)
                )
                this_expert_bias_down = nn.Parameter(
                    torch.empty((out_features,), **factory_kwargs)
                )
                self.bias_gate.append(this_expert_bias_gate)
                self.bias_up.append(this_expert_bias_up)
                self.bias_down.append(this_expert_bias_down)
        else:
            self.register_parameter("bias_gate", None)
            self.register_parameter("bias_up", None)
            self.register_parameter("bias_down", None)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight_gate[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_up[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_down[i], a=math.sqrt(5))
            if self.bias_gate is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_gate[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_gate[i], -bound, bound)
            if self.bias_up is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_up[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_up[i], -bound, bound)
            if self.bias_down is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_down[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_down[i], -bound, bound)

    def forward(self, input, i):
        gate = self.act_fn(
            F.linear(
                input,
                self.weight_gate[i],
                self.bias_gate[i] if self.bias_gate is not None else None,
            )
        )
        up = F.linear(
            input,
            self.weight_up[i],
            self.bias_up[i] if self.bias_up is not None else None,
        )
        down = F.linear(
            gate * up,
            self.weight_down[i],
            self.bias_down[i] if self.bias_down is not None else None,
        )
        return down

    def extra_repr(self):
        return (
            "in_features={}, hidden_features={}, out_features={}, hidden_act={},"
            " num_experts={}, size_experts={}, bias={}".format(
                self.in_features,
                self.hidden_features,
                self.out_features,
                self.hidden_act,
                self.num_experts,
                self.size_experts,
                self.bias_gate is not None,
            )
        )


class UniversalCalculator(nn.Module):
    def __init__(
        self,
        experts: LinearGLUExperts,
        multiply_gate_scores=True,
        score_scale_factor=1.0,
        add_weight_norm: bool = False,
    ):
        super(UniversalCalculator, self).__init__()
        self.experts = experts
        # TODO (zhutong): use vmap to boost the training efficiency
        # self.experts_vmap = torch.vmap(self.experts)
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.num_experts = experts.num_experts
        self.mlp_norm = None
        if multiply_gate_scores and add_weight_norm:
            raise NotImplementedError

    def reset_experts(self):
        self.experts.reset_parameters()

    def forward(
        self, x, topK_indices, topK_scores, expert_batch_size=None, **kwargs
    ) -> CalculatorOutput:
        batch_size = topK_indices.size(0)  # topK_indices: (bsz*seq_len, num_selects)
        num_selects = topK_indices.size(1)
        topK_indices = topK_indices.flatten()  # shape(batch_size*num_selects)
        topK_scores = topK_scores.flatten()  # shape(batch_size*num_selects)
        batch_indices = torch.arange(
            batch_size, device=topK_scores.device
        ).repeat_interleave(num_selects)

        _, index_sorted_topK_indices = topK_indices.sort(0)

        sorted_topK_scores = topK_scores.index_select(0, index_sorted_topK_indices)
        sorted_batch_indices = batch_indices.index_select(0, index_sorted_topK_indices)

        if expert_batch_size is None:
            expert_batch_size = topK_indices.bincount(
                minlength=self.num_experts
            ).tolist()

        sorted_x = x.index_select(0, sorted_batch_indices)
        split_x = torch.split(sorted_x, expert_batch_size, dim=0)

        expert_outputs = [
            self.experts(split_x[i], i)
            for i in range(self.num_experts)
            if split_x[i].shape[0] > 0
        ]

        # (bsz*seq_len*num_selects, hidden_size)
        cat_expert_outputs = torch.cat(expert_outputs, 0)
        output_dim = cat_expert_outputs.size(1)
        if self.multiply_gate_scores:
            if self.mlp_norm is None:
                cat_expert_outputs = torch.mul(
                    cat_expert_outputs,
                    sorted_topK_scores.reshape(-1, 1) * self.score_scale_factor,
                )
                # cat_expert_outputs = torch.mul(cat_expert_outputs, sorted_topK_scores.reshape(-1, 1) * 1.0)
            else:
                cat_expert_outputs = torch.mul(
                    cat_expert_outputs, sorted_topK_scores.reshape(-1, 1)
                )
                cat_expert_outputs = self.mlp_norm(cat_expert_outputs)

        zeros = torch.zeros(
            (batch_size, output_dim),
            device=cat_expert_outputs.device,
            dtype=cat_expert_outputs.dtype,
        )
        y = zeros.index_add(0, sorted_batch_indices, cat_expert_outputs)

        return CalculatorOutput(hidden_states=y, num_dropped_tokens=torch.tensor(-1.0))


class BaseMoELayer(nn.Module):
    def __init__(self):
        super(BaseMoELayer, self).__init__()

        self.gate: TopKBalancedNoisyGate
        self.calculator: UniversalCalculator

    def _create_gate(self, **kwargs):
        self.gate_type = kwargs.get("gate_type", "TopKBalancedNoisyGate")

        if self.gate_type == "TopKBalancedNoisyGate":  # noisy gate
            self.gate = TopKBalancedNoisyGate(
                self.input_size,
                self.num_experts,
                self.num_selects,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", True),
                use_balance=kwargs.get("gate_use_balance", True),
                balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
                add_noise=kwargs.get("gate_add_noise", True),
                noise_epsilon=kwargs.get("gate_noise_epsilon", 1e-2),
            )
        else:
            raise NotImplementedError

    def _create_calculator(self, experts, **kwargs):
        self.calculator_type = kwargs.get("calculator_type", "UniversalCalculator")

        if self.calculator_type == "UniversalCalculator":  # top K calculator
            self.calculator = UniversalCalculator(
                experts,
                multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
                score_scale_factor=kwargs.get("score_scale_factor", 1.0),
                add_weight_norm=kwargs.get("add_weight_norm", False),
            )
        else:
            raise NotImplementedError

    def forward(self, x, attention_mask=None) -> MoEMlpOutput:
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_size)
        flattened_mask = None
        if attention_mask is not None and len(attention_mask.shape) == 2:
            flattened_mask = attention_mask.flatten()
            flattened_shape = flattened_mask.shape
            x = x[flattened_mask.bool()]

        gate_outputs: dict = self.gate(x)
        calc_outs: CalculatorOutput = self.calculator(x, **gate_outputs)

        y = calc_outs.hidden_states
        expert2tokens = None
        if flattened_mask is not None:
            y = torch.zeros(
                flattened_shape + (self.output_size,), dtype=x.dtype, device=x.device
            )  # (batch_size*seq_len, output_size)
            y[
                flattened_mask.bool()
            ] = calc_outs.hidden_states  # (non_padding_num, output_size)

            # # for stats only
            # expert2tokens = {e: set() for e in range(self.gate.num_experts)}
            # i = 0
            # for t, m in enumerate(flattened_mask.tolist()):
            #     if m:
            #         selected_experts = gate_outputs["topK_indices"][i].tolist()
            #         for e in selected_experts:
            #             expert2tokens[e].add(t)
            #         i += 1

        y = y.reshape(original_shape + (self.output_size,))

        return MoEMlpOutput(
            hidden_states=y,
            balance_loss=gate_outputs.get("balance_loss"),
            num_dropped_tokens=calc_outs.num_dropped_tokens,
            gate_load=gate_outputs.get("load", torch.tensor(-1)),
            gate_importance=gate_outputs.get("importance", torch.tensor(-1)),
            expert2tokens=expert2tokens,
        )

    def reset_gate_network(self):
        self.gate.reset_gate_network()

    def reset_experts(self):
        self.calculator.reset_experts()


class LinearGLUMoELayer(BaseMoELayer):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        hidden_act,
        num_experts,
        num_selects,
        size_experts=None,
        bias=True,
        **kwargs,
    ):
        super(LinearGLUMoELayer, self).__init__()
        assert num_selects <= num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_act = hidden_act
        self.num_experts = num_experts
        self.num_selects = num_selects
        self.size_experts = size_experts
        self.bias = bias

        experts = LinearGLUExperts(
            input_size,
            hidden_size,
            output_size,
            hidden_act,
            num_experts,
            size_experts=size_experts,
            bias=bias,
        )

        self._create_gate(**kwargs)
        self._create_calculator(experts, **kwargs)


class LlamaMoEDecoderLayer(nn.Module):
    def __init__(self, config: LlamaMoEConfig, layer_index):
        super().__init__()

        self.hidden_size = config.hidden_size
        # self.self_attn = LlamaAttention(config=config)
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_index
        )

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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
            "score_scale_factor": (
                config.score_scale_factor[layer_index]
                if isinstance(config.score_scale_factor, list)
                else config.score_scale_factor
            ),
            "add_weight_norm": config.add_weight_norm,
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
    ) -> tuple:
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
        mlp_outs: MoEMlpOutput = self.mlp(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + mlp_outs.hidden_states

        outputs = (
            hidden_states,
            mlp_outs.balance_loss,
            mlp_outs.num_dropped_tokens,
            mlp_outs.gate_load,
            mlp_outs.gate_importance,
            mlp_outs.expert2tokens,
        )
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaMoEPreTrainedModel(PreTrainedModel):
    config_class = LlamaMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaMoEDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LlamaMoEModel(LlamaMoEPreTrainedModel):
    def __init__(self, config: LlamaMoEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [LlamaMoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds
        balance_loss = 0.0

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        num_dropped_tokens = ()
        gate_load = ()
        gate_importance = ()
        expert2tokens = ()
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if layer_outputs[1] is not None:
                balance_loss += layer_outputs[1]

            if use_cache:
                next_decoder_cache = layer_outputs[6 if output_attentions else 5]

            if output_attentions:
                all_self_attns += (layer_outputs[5],)

            num_dropped_tokens += (layer_outputs[2],)
            gate_load += (layer_outputs[3],)
            gate_importance += (layer_outputs[4],)
            expert2tokens += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
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
            expert2tokens=expert2tokens,
        )

    def reset_gate_network(self):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.reset_gate_network()

    def reset_experts(self):
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer.reset_experts()


class LlamaMoEForCausalLM(LlamaMoEPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaMoEModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

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
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if outputs.balance_loss is not None and outputs.balance_loss > 0:
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
            expert2tokens=outputs.expert2tokens,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    def reset_gate_network(self):
        self.model.reset_gate_network()

    def reset_experts(self):
        self.model.reset_experts()
