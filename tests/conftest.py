import json

import pytest
import torch
from safetensors.torch import save_file

from rover.models.qwen3.config import Qwen3Config
from rover.models.qwen3.text import Qwen3Model
from rover.models.qwen3.vision import Qwen3VisionModel
from rover.runtime.config import DEFAULT_SAMPLING_PARAMS, RuntimeConfig


@pytest.fixture
def tiny_model(tmp_path):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    config = Qwen3Config(
        text_config=dict(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            eos_token_id=-1,
            rope_parameters={"rope_theta": 10000, "mrope_section": [4, 2, 2]},
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0],
        ),
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=28,
        tie_word_embeddings=False,
    )
    runtime = RuntimeConfig(
        model_dir=str(model_dir),
        device="cpu",
        state_dir=None,
        stop_after_tokens=None,
        max_decode_tokens=5,
        max_cache_len=32,
        max_batch_size=1,
        prefill_chunk_size=2,
        prefill_order="chunk_first",
        sampling=DEFAULT_SAMPLING_PARAMS,
    )
    text = Qwen3Model(config, runtime, None).to(dtype=torch.bfloat16)
    vision = Qwen3VisionModel(config.vision_config).to(dtype=torch.bfloat16)
    weights = {"model.language_model." + k: v.contiguous() for k, v in text.state_dict().items()}
    weights.update({"model.visual." + k: v.contiguous() for k, v in vision.state_dict().items()})
    weights["lm_head.weight"] = torch.randn(32, 32, dtype=torch.bfloat16) * 0.1
    save_file(weights, str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(json.dumps(config.to_dict()))
    return model_dir
