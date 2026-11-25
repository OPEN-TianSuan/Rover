"""Visual input placement and multimodal token positions."""

from dataclasses import dataclass

import torch

from rover.runtime.state import GenerationState


@dataclass
class VisualInput:
    token_id: int
    pixel_values: torch.Tensor
    vision_grid_thw: torch.Tensor
    llm_grid_thw: torch.Tensor


def build_visual_inputs(state: GenerationState) -> list[VisualInput]:
    visual_inputs = []
    if state.pixel_values is not None:
        assert state.image_grid_thw is not None
        visual_inputs.append(
            VisualInput(
                token_id=state.config.image_token_id,
                pixel_values=state.pixel_values,
                vision_grid_thw=state.image_grid_thw,
                llm_grid_thw=state.image_grid_thw,
            )
        )
    if state.pixel_values_videos is not None:
        assert state.video_grid_thw is not None
        llm_grid_thw = torch.repeat_interleave(state.video_grid_thw, state.video_grid_thw[:, 0], dim=0).clone()
        llm_grid_thw[:, 0] = 1
        visual_inputs.append(
            VisualInput(
                token_id=state.config.video_token_id,
                pixel_values=state.pixel_values_videos,
                vision_grid_thw=state.video_grid_thw,
                llm_grid_thw=llm_grid_thw,
            )
        )
    return visual_inputs


def compute_rope_index(
    input_ids: torch.Tensor,
    spatial_merge_size: int,
    vision_start_token_id: int,
    visual_inputs: list[VisualInput],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute multimodal RoPE indices for a text sequence with image/video placeholders.

    Args:
        input_ids: Token ids of shape `(batch_size, sequence_length)`.
        spatial_merge_size: Spatial merge factor used to map vision-grid coordinates to LLM token coordinates.
        vision_start_token_id: Token id that marks the start of a visual block in `input_ids`.
        visual_inputs: Visual metadata for each modality. `visual_input.llm_grid_thw` is consumed in prompt order.
        attention_mask: Optional mask of shape `(batch_size, sequence_length)`. Positions with value 0 are skipped.

    Returns:
        A tuple `(position_ids, mrope_position_deltas)`.

        - `position_ids`: Long tensor of shape `(3, batch_size, sequence_length)`.
          The first dimension stores temporal, height, and width RoPE indices.
        - `mrope_position_deltas`: Tensor of shape `(batch_size, 1)`.
          For each sample, this is `max(position_ids) + 1 - sequence_length` and is used to offset decode positions.
    """
    visual_grid_lists = {visual_input.token_id: visual_input.llm_grid_thw.tolist() for visual_input in visual_inputs}
    visual_indices = {visual_input.token_id: 0 for visual_input in visual_inputs}

    position_ids = torch.zeros(
        3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
    )
    mrope_position_deltas = []

    for i, ids in enumerate(input_ids):
        input_tokens = ids.tolist()
        vision_start_indices = torch.argwhere(ids == vision_start_token_id).squeeze(1)
        vision_tokens = ids[vision_start_indices + 1] if vision_start_indices.numel() > 0 else ids.new_empty(0)

        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        for token_id in vision_tokens.tolist():
            ed = input_tokens.index(token_id, st)
            t, h, w = visual_grid_lists[token_id][visual_indices[token_id]]
            visual_indices[token_id] += 1

            llm_grid_t, llm_grid_h, llm_grid_w = t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            if text_len > 0:
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                )

            t_index = (
                torch.arange(llm_grid_t, device=input_ids.device)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h, device=input_ids.device)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w, device=input_ids.device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            st_idx_vis = st_idx + text_len
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx_vis)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, :] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append((llm_positions.max() + 1 - len(input_ids[i])).item())

    mrope_position_deltas_tensor = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas_tensor
