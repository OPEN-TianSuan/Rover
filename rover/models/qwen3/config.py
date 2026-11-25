"""Qwen3 model configuration"""

import logging
from typing import Any

from rover.models.config import Config

RopeParameters = dict[str, Any]

logger = logging.getLogger(__name__)


class Qwen3VisionConfig(Config):
    model_type = "qwen3_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=24,
        hidden_size=1024,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4096,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=2048,
        num_position_embeddings=2304,
        deepstack_visual_indexes=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = (
            list(deepstack_visual_indexes) if deepstack_visual_indexes is not None else [5, 11, 17]
        )


class Qwen3TextConfig(Config):
    model_type = "qwen3_vl_text"
    base_config_key = "text_config"
    default_theta = 5000000.0

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = dict(rope_parameters or kwargs.pop("rope_scaling", None) or {})
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", self.default_theta))
        self.rope_parameters.setdefault("mrope_section", [24, 20, 20])
        self.pad_token_id = pad_token_id

        super().__init__(
            **kwargs,
        )


class Qwen3Config(Config):
    model_type = "qwen3_vl"
    sub_configs = {"vision_config": Qwen3VisionConfig, "text_config": Qwen3TextConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        else:
            self.vision_config = vision_config if vision_config is not None else self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        else:
            self.text_config = text_config if text_config is not None else self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)

    def validate(self):
        tc = self.text_config
        assert tc.rope_parameters is not None
        assert "rope_theta" in tc.rope_parameters
        assert "mrope_section" in tc.rope_parameters


__all__ = ["Qwen3Config", "Qwen3TextConfig", "Qwen3VisionConfig"]
