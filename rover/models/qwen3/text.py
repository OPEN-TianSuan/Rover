"""Qwen3 language model layers."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.bias import causal_lower_right

from rover.cache.dense import dense_attention_forward
from rover.cache.kivi import kivi_attention_forward
from rover.cache.turboquant import (
    LayerTurboQuantCache,
    TurboQuantGraphBuffers,
    _graph_write_store,
    turboquant_attention_forward,
    turboquant_attention_graph_forward,
)
from rover.kernels import loader as _ext
from rover.models.quantization import GptqTextConfig, build_text_linear
from rover.models.qwen3.config import Qwen3Config
from rover.nn.attention import repeat_kv
from rover.nn.modules import ACT2FN, InferenceModule
from rover.nn.rotary import RotaryEmbedding
from rover.runtime.config import RuntimeConfig
from rover.runtime.state import GenerationState
from rover.runtime.types import Phase, Range


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(hidden_states, self.weight.shape, self.weight, self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config, text_quant_config: GptqTextConfig | None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = build_text_linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            text_quant_config=text_quant_config,
        )
        self.up_proj = build_text_linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            text_quant_config=text_quant_config,
        )
        self.down_proj = build_text_linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            text_quant_config=text_quant_config,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Qwen3Config,
        runtime_config: RuntimeConfig,
        layer_idx: int,
        text_quant_config: GptqTextConfig | None,
    ):
        super().__init__()
        self.config = config
        self.runtime_config = runtime_config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self._device = torch.device(runtime_config.device)

        q_dim = config.num_attention_heads * self.head_dim
        kv_dim = config.num_key_value_heads * self.head_dim

        self.q_proj = build_text_linear(
            config.hidden_size,
            q_dim,
            bias=config.attention_bias,
            text_quant_config=text_quant_config,
        )
        self.k_proj = build_text_linear(
            config.hidden_size,
            kv_dim,
            bias=config.attention_bias,
            text_quant_config=text_quant_config,
        )
        self.v_proj = build_text_linear(
            config.hidden_size,
            kv_dim,
            bias=config.attention_bias,
            text_quant_config=text_quant_config,
        )
        self.o_proj = build_text_linear(
            q_dim,
            config.hidden_size,
            bias=config.attention_bias,
            text_quant_config=text_quant_config,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _eager_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
    ):
        key_states = repeat_kv(key, self.num_key_value_groups)
        value_states = repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    def repeat_kv_for_decode(self, states: torch.Tensor) -> torch.Tensor:
        return repeat_kv(states, self.num_key_value_groups)

    def _build_attention_mask(self, r: Range) -> torch.Tensor:
        q_positions = torch.arange(r.start, r.end, device=self._device).unsqueeze(-1)
        k_positions = torch.arange(r.end, device=self._device).unsqueeze(0)
        visible = k_positions <= q_positions

        bias = torch.zeros((1, 1, r.length, r.end), dtype=torch.float32, device=self._device)
        return bias.masked_fill(~visible, float("-inf"))

    def _build_attention_bias(self, r: Range):
        return causal_lower_right(r.length, r.end)

    def forward(
        self,
        hidden_states: torch.Tensor,
        r: Range,
        pos_emb: RotaryEmbedding,
        generation_state: GenerationState,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = pos_emb.apply(query_states, key_states)

        generation_state.update_kv(r, key_states, value_states)

        if self.runtime_config.kv_cache_format == "kivi":
            layer_cache = generation_state.get_kv_for_attention(r.layer)
            attention_mask = self._build_attention_mask(r) if r.length > 1 else None
            attn_output, attn_weights = kivi_attention_forward(
                self,
                query_states,
                layer_cache,
                attention_mask,
                scaling=self.scaling,
                num_key_value_groups=self.num_key_value_groups,
            )
        elif self.runtime_config.kv_cache_format == "turboquant" and generation_state.phase == Phase.DECODING:
            layer_cache = generation_state.get_kv_for_attention(r.layer)
            attn_output, attn_weights = turboquant_attention_forward(
                query_states,
                layer_cache,
                self._build_attention_mask(r),
                scaling=self.scaling,
            )
        else:
            attn_output, attn_weights = dense_attention_forward(self, generation_state, r, query_states)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _graph_forward(
        self,
        hidden_states: torch.Tensor,
        pos_emb: RotaryEmbedding,
        write_idx_exp: torch.Tensor,
        key_cache: torch.Tensor,
        val_cache: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode attention step suitable for CUDA graph capture.

        Args:
            write_idx_exp: (1, kv_heads, 1, head_dim) int64, scatter index updated before each replay.
            key_cache / val_cache: full (1, kv_heads, max_len, head_dim) KV buffers for this layer.
            attn_mask: (1, 1, 1, max_len) float32, 0 for valid positions and -inf for masked.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        val = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query, key = pos_emb.apply(query, key)

        # Scatter-write K/V into cache at the current decode position (graph-safe).
        key_cache.scatter_(2, write_idx_exp, key)
        val_cache.scatter_(2, write_idx_exp, val)

        # Full-buffer attention with pre-allocated causal mask.
        k_full = self.repeat_kv_for_decode(key_cache)
        v_full = self.repeat_kv_for_decode(val_cache)
        scores = torch.matmul(query, k_full.transpose(2, 3)).mul_(self.scaling).add_(attn_mask)
        weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(weights, v_full).transpose(1, 2).contiguous()

        return self.o_proj(out.reshape(*input_shape, -1).contiguous())

    def _graph_forward_turboquant(
        self,
        hidden_states: torch.Tensor,
        pos_emb: RotaryEmbedding,
        layer_cache: LayerTurboQuantCache,
        start_tensor: torch.Tensor,
        start_index_tensor: torch.Tensor,
        attn_mask: torch.Tensor,
        bufs: TurboQuantGraphBuffers,
        total_len_tensor: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        val = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query, key = pos_emb.apply(query, key)

        if layer_cache.config.mode == "mse":
            kr, ko = layer_cache.key_regular, layer_cache.key_outlier
            vr, vo = layer_cache.value_regular, layer_cache.value_outlier
            _ext.append_kv_mse_split_graph(
                key[0].to(torch.bfloat16).contiguous(),
                val[0].to(torch.bfloat16).contiguous(),
                layer_cache.regular_idx_i32,
                layer_cache.outlier_idx_i32,
                kr.quantizer.rotation_t,
                ko.quantizer.rotation_t,
                kr.quantizer.boundaries.contiguous(),
                ko.quantizer.boundaries.contiguous(),
                vr.quantizer.rotation_t,
                vo.quantizer.rotation_t,
                vr.quantizer.boundaries.contiguous(),
                vo.quantizer.boundaries.contiguous(),
                kr._packed,
                kr._norms[:, :, 0],
                kr._norms_f32[:, :, 0],
                ko._packed,
                ko._norms[:, :, 0],
                ko._norms_f32[:, :, 0],
                vr._packed,
                vr._norms[:, :, 0],
                vr._norms_f32[:, :, 0],
                vo._packed,
                vo._norms[:, :, 0],
                vo._norms_f32[:, :, 0],
                start_tensor,
                int(kr.quantizer.bits),
                int(ko.quantizer.bits),
            )
        else:
            key_bf16 = key[0].to(torch.bfloat16).contiguous()
            val_bf16 = val[0].to(torch.bfloat16).contiguous()
            _graph_write_store(
                layer_cache.key_regular,
                bufs.prod_key_regular,
                key_bf16.index_select(-1, layer_cache.regular_idx),
                start_index_tensor,
            )
            _graph_write_store(
                layer_cache.value_regular,
                bufs.prod_value_regular,
                val_bf16.index_select(-1, layer_cache.regular_idx),
                start_index_tensor,
            )
            _graph_write_store(
                layer_cache.key_outlier,
                bufs.prod_key_outlier,
                key_bf16.index_select(-1, layer_cache.outlier_idx),
                start_index_tensor,
            )
            _graph_write_store(
                layer_cache.value_outlier,
                bufs.prod_value_outlier,
                val_bf16.index_select(-1, layer_cache.outlier_idx),
                start_index_tensor,
            )

        attn_output = turboquant_attention_graph_forward(
            query,
            layer_cache,
            attn_mask,
            self.scaling,
            bufs,
            total_len_tensor,
            max_len,
        )
        return self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        runtime_config: RuntimeConfig,
        layer_idx: int,
        text_quant_config: GptqTextConfig | None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(
            config=config,
            runtime_config=runtime_config,
            layer_idx=layer_idx,
            text_quant_config=text_quant_config,
        )
        self.mlp = Qwen3MLP(config, text_quant_config=text_quant_config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        r: Range,
        pos_emb: RotaryEmbedding,
        generation_state: GenerationState,
    ) -> torch.Tensor:
        assert r.length == hidden_states.shape[-2]
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            r=r,
            pos_emb=pos_emb,
            generation_state=generation_state,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def _graph_forward(
        self,
        hidden_states: torch.Tensor,
        pos_emb: RotaryEmbedding,
        write_idx_exp: torch.Tensor,
        key_cache: torch.Tensor,
        val_cache: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn._graph_forward(
            hidden_states, pos_emb, write_idx_exp, key_cache, val_cache, attn_mask
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def _graph_forward_turboquant(
        self,
        hidden_states: torch.Tensor,
        pos_emb: RotaryEmbedding,
        layer_cache: LayerTurboQuantCache,
        start_tensor: torch.Tensor,
        start_index_tensor: torch.Tensor,
        attn_mask: torch.Tensor,
        bufs: TurboQuantGraphBuffers,
        total_len_tensor: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn._graph_forward_turboquant(
            hidden_states,
            pos_emb,
            layer_cache,
            start_tensor,
            start_index_tensor,
            attn_mask,
            bufs,
            total_len_tensor,
            max_len,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3Model(InferenceModule):
    def __init__(
        self,
        config: Qwen3Config,
        runtime_config: RuntimeConfig,
        text_quant_config: GptqTextConfig | None,
    ):
        super().__init__(config)
        tc = config.text_config
        self.padding_idx = tc.pad_token_id
        self.vocab_size = tc.vocab_size

        self.embed_tokens = nn.Embedding(tc.vocab_size, tc.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    tc,
                    runtime_config,
                    layer_idx,
                    text_quant_config=text_quant_config,
                )
                for layer_idx in range(tc.num_hidden_layers)
            ]
        )
        self.runtime_config = runtime_config
        self.norm = Qwen3RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
        self.gradient_checkpointing = False
        self.has_sliding_layers = False
