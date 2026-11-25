"""Qwen3 visual encoder and feature merging."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from rover.diagnostics.logging import logger
from rover.diagnostics.tracing import tracer
from rover.models.quantization import GptqTextConfig, build_visual_linear
from rover.models.qwen3.position import build_vision_rotary_inv_freq
from rover.nn.modules import ACT2FN, InferenceModule, VisionOutput
from rover.nn.rotary import RotaryEmbedding


def vision_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=False,
    )
    return attn_output.transpose(1, 2).contiguous(), None


@dataclass
class BaseModelOutputWithDeepstackFeatures(VisionOutput):
    deepstack_features: list[torch.FloatTensor] | None = None


class Qwen3VisionMLP(nn.Module):
    def __init__(self, config, visual_quant_config: GptqTextConfig | None = None):
        super().__init__()
        self.linear_fc1 = build_visual_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            visual_quant_config=visual_quant_config,
        )
        self.linear_fc2 = build_visual_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            visual_quant_config=visual_quant_config,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


class Qwen3VisionPatchMerger(nn.Module):
    def __init__(
        self,
        config,
        use_postshuffle_norm=False,
        visual_quant_config: GptqTextConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = build_visual_linear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            visual_quant_config=visual_quant_config,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = build_visual_linear(
            self.hidden_size,
            config.out_hidden_size,
            bias=True,
            visual_quant_config=visual_quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VisionAttention(nn.Module):
    def __init__(self, config, visual_quant_config: GptqTextConfig | None = None) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_key_value_groups = 1
        self.qkv = build_visual_linear(
            config.hidden_size,
            config.hidden_size * 3,
            bias=True,
            visual_quant_config=visual_quant_config,
        )
        self.proj = build_visual_linear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            visual_quant_config=visual_quant_config,
        )
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.is_causal = False

    def forward(self, hidden_states, cu_seqlens, pos_emb: RotaryEmbedding):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 2, 0, 3).unbind(0)
        )
        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)

        query_states, key_states = pos_emb.apply(query_states, key_states)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [torch.split(tensor, lengths, dim=2) for tensor in (query_states, key_states, value_states)]
        attn_outputs = [
            vision_eager_attention_forward(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                is_causal=False,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Qwen3VisionBlock(nn.Module):
    def __init__(self, config, visual_quant_config: GptqTextConfig | None = None) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VisionAttention(config=config, visual_quant_config=visual_quant_config)
        self.mlp = Qwen3VisionMLP(config=config, visual_quant_config=visual_quant_config)

    def forward(self, hidden_states, cu_seqlens, pos_emb: RotaryEmbedding):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens=cu_seqlens, pos_emb=pos_emb)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VisionModel(InferenceModule):
    def __init__(self, config, visual_quant_config: GptqTextConfig | None = None) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        self.register_buffer("rotary_inv_freq", build_vision_rotary_inv_freq(config), persistent=False)

        self.blocks = nn.ModuleList(
            [Qwen3VisionBlock(config, visual_quant_config=visual_quant_config) for _ in range(config.depth)]
        )
        self.merger = Qwen3VisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
            visual_quant_config=visual_quant_config,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                    visual_quant_config=visual_quant_config,
                )
                for _ in config.deepstack_visual_indexes
            ]
        )

        self.gradient_checkpointing = False

    def prepare_stem(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        stem_timer = tracer.timer("vision-stem").start()
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)
        stem_timer.end()
        logger.info("[vision] stem %.2fms", stem_timer.elapsed_ms())
        return hidden_states

    def prepare_layout(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, RotaryEmbedding]:
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        pos_emb = RotaryEmbedding.from_freqs(rotary_pos_emb.unsqueeze(0))
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        return cu_seqlens, pos_emb

    def run_layer(
        self,
        layer_num: int,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        pos_emb: RotaryEmbedding,
    ) -> torch.Tensor:
        layer_timer = tracer.timer("vision-layer", layer_num=layer_num).start()
        hidden_states = self.blocks[layer_num](hidden_states, cu_seqlens=cu_seqlens, pos_emb=pos_emb)
        if hidden_states.is_cuda:
            torch.cuda.synchronize(hidden_states.device)
        layer_timer.end()
        logger.info(
            "[vision] layer %d/%d %.2fms",
            layer_num + 1,
            len(self.blocks),
            layer_timer.elapsed_ms(),
        )
        return hidden_states

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Build 2D rotary position frequencies for all visual patches in prompt order.

        Args:
            grid_thw: Long tensor of shape `(num_visual_items, 3)`. Each row is `[t, h, w]` in the
                vision encoder grid before spatial merge.

        Returns:
            Tensor of shape `(sum(t * h * w), head_dim // 2)`.
            Rows are aligned with the flattened patch order consumed by the vision transformer.
        """
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        seq = torch.arange(max_hw, device=self.rotary_inv_freq.device, dtype=self.rotary_inv_freq.dtype)
        freq_table = torch.outer(seq, self.rotary_inv_freq)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = (
                (block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None])
                .expand(merged_h, merged_w, merge_size, merge_size)
                .reshape(-1)
            )
            col_idx = (
                (block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :])
                .expand(merged_h, merged_w, merge_size, merge_size)
                .reshape(-1)
            )
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        return freq_table[pos_ids].flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Interpolate learned 2D positional embeddings to the visual grids in `grid_thw`.

        Args:
            grid_thw: Long tensor of shape `(num_visual_items, 3)`. Each row is `[t, h, w]` in the
                vision encoder grid before spatial merge.

        Returns:
            Tensor of shape `(sum(t * h * w), hidden_size)`.
            The output is flattened in the same order as the vision patches after expanding each item's
            interpolated 2D grid across its temporal dimension.
        """
        grid_thw_list = grid_thw.tolist()
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds.sum(0)

        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        merge_size = self.spatial_merge_size
        result = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            result.append(pos_embed)
        return torch.cat(result)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> BaseModelOutputWithDeepstackFeatures:
        hidden_states = self.prepare_stem(hidden_states, grid_thw)
        cu_seqlens, pos_emb = self.prepare_layout(grid_thw)
        deepstack_features = []
        for layer_num in range(len(self.blocks)):
            hidden_states = self.run_layer(layer_num, hidden_states, cu_seqlens, pos_emb)
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(self.deepstack_merger_list[idx](hidden_states))

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=self.merger(hidden_states),
            deepstack_features=deepstack_features,
        )
