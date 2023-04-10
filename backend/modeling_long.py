import math
import torch
from flash_attn.modules.mha import FlashSelfAttention
from typing import Optional, Tuple
from transformers import GPTNeoXForCausalLM, LlamaForCausalLM
from transformers.models.gpt_neox import modeling_gpt_neox
from transformers.models.llama import modeling_llama
from torch import nn


class GPTNeoXFlashAttentionWrapper(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.flash_self_attention = FlashSelfAttention(
            causal=True, softmax_scale=1/self.attention.norm_factor)
        self.dropout_p = 0.0

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                layer_past=None,
                use_cache=False,
                output_attentions=False):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.attention.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size(
        )[:-1] + (self.attention.num_attention_heads, 3 * self.attention.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.attention.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.attention.head_size: 2 *
                  self.attention.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.attention.head_size:].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.attention.rotary_ndims]
        query_pass = query[..., self.attention.rotary_ndims:]
        key_rot = key[..., : self.attention.rotary_ndims]
        key_pass = key[..., self.attention.rotary_ndims:]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        cos, sin = self.attention.rotary_emb(value, seq_len=seq_len)
        query, key = modeling_gpt_neox.apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, offset=offset)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        # attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        qkv = torch.concat([query.unsqueeze(2), key.unsqueeze(
            2), value.unsqueeze(2)], dim=2).permute(0, 3, 2, 1, 4).half()
        attn_output = self.flash_self_attention(qkv)
        attn_weights = None

        # Reshape outputs
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(
            1), self.attention.num_attention_heads * self.attention.head_size)
        attn_output = self.attention.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GPTNeoXLongForCausalLM(GPTNeoXForCausalLM):
    def __init__(self, config):
        super(modeling_gpt_neox.GPTNeoXPreTrainedModel, self).__init__(config)

        self.gpt_neox = modeling_gpt_neox.GPTNeoXModel(config)
        self.embed_out = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        for each in self.gpt_neox.layers:
            # original_emb = each.attention.rotary_emb
            each.attention.rotary_emb = modeling_gpt_neox.RotaryEmbedding(
                each.attention.rotary_ndims, config.max_positions, 10000)
            each.attention.bias = torch.tril(torch.ones((config.max_positions, config.max_positions), dtype=torch.uint8)).view(
                1, 1, config.max_positions, config.max_positions
            )
            each.attention = GPTNeoXFlashAttentionWrapper(each.attention)

        # Initialize weights and apply final processing
        self.post_init()


class LlamaFlashAttentionWrapper(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.flash_self_attention = FlashSelfAttention(
            causal=True, softmax_scale=1/math.sqrt(self.attention.head_dim))
        self.dropout_p = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_ids=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.size()

        query_states = self.attention.q_proj(hidden_states).view(
            bsz, q_len, self.attention.num_heads, self.attention.head_dim).transpose(1, 2)
        key_states = self.attention.k_proj(hidden_states).view(
            bsz, q_len, self.attention.num_heads, self.attention.head_dim).transpose(1, 2)
        value_states = self.attention.v_proj(hidden_states).view(
            bsz, q_len, self.attention.num_heads, self.attention.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset
        cos, sin = self.attention.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = modeling_llama.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, offset=offset)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # attn_weights = torch.matmul(
        #     query_states, key_states.transpose(2, 3)) / math.sqrt(self.attention.head_dim)

        # if attn_weights.size() != (bsz, self.attention.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz * self.attention.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights + attention_mask
        #     attn_weights = torch.max(attn_weights, torch.tensor(
        #         torch.finfo(attn_weights.dtype).min))

        # # upcast attention to fp32
        # attn_weights = nn.functional.softmax(
        #     attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_output = torch.matmul(attn_weights, value_states)

        # if attn_output.size() != (bsz, self.attention.num_heads, q_len, self.attention.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.attention.num_heads, q_len, self.attention.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        qkv = torch.concat([query_states.unsqueeze(2), key_states.unsqueeze(
            2), value_states.unsqueeze(2)], dim=2).permute(0, 3, 2, 1, 4).to(query_states.dtype)
        attn_output = self.flash_self_attention(qkv)
        attn_weights = None

        # attn_output = attn_output.transpose(1, 2)
        # attn_output = attn_output.reshape(
        #     bsz, q_len, self.attention.hidden_size)

        attn_output = attn_output.view(attn_output.size(0), attn_output.size(
            1), self.attention.num_heads * self.attention.head_dim)

        attn_output = self.attention.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaLongForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super(modeling_llama.LlamaPreTrainedModel, self).__init__(config)
        self.model = modeling_llama.LlamaModel(config)

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        for each in self.model.layers:
            each.self_attn.rotary_emb = modeling_llama.LlamaRotaryEmbedding(
                each.self_attn.head_dim, config.max_positions, 10000)
            each.self_attn.bias = torch.tril(torch.ones((config.max_positions, config.max_positions), dtype=torch.uint8)).view(
                1, 1, config.max_positions, config.max_positions
            )
            each.self_attn = LlamaFlashAttentionWrapper(each.self_attn)

        # Initialize weights and apply final processing
        self.post_init()
