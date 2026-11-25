import json
import subprocess
import sys

import pytest
import torch

from rover.engine import Engine
from rover.models.qwen3.config import Qwen3Config, Qwen3TextConfig


def test_core_import_without_processing():
    code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"tokenizers", "torchvision", "jinja2"}:
            raise ImportError("Input processing must remain separate from model execution")
sys.meta_path.insert(0, Block())
from rover.engine import Engine
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_config_roundtrip():
    text = Qwen3TextConfig(rope_scaling={"mrope_section": [24, 20, 20]}, rope_theta=5000000)
    config = Qwen3Config(text_config=text)
    restored = Qwen3Config.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored.text_config.rope_parameters == text.rope_parameters
    assert config.text_config is text


@pytest.mark.parametrize(
    "order,strategy",
    [
        ("chunk_first", "chunk_first"),
        ("layer_first", "chunk_first"),
        ("scheduled", "chunk_first"),
        ("scheduled", "layer_first"),
        ("scheduled", "greedy"),
        ("scheduled", "state_compression_dp"),
    ],
)
def test_generation_and_resume(tiny_model, tmp_path, order, strategy):
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with Engine.create(tiny_model, tmp_path / "reference", ids, max_new_tokens=5, chunk_size=8) as engine:
        expected = engine.generate()
    state_dir = tmp_path / "session"
    with Engine.create(
        tiny_model,
        state_dir,
        ids,
        max_new_tokens=5,
        chunk_size=2,
        prefill_order=order,
        strategy=strategy,
        stop_after_tokens=2,
    ) as engine:
        assert engine.generate() == expected[:2]
    with Engine.resume(state_dir) as engine:
        assert engine.generate() == expected
    with pytest.raises(FileExistsError):
        Engine.create(tiny_model, state_dir, ids)


def test_multimodal_resume(tiny_model, tmp_path):
    ids = torch.tensor([[1, 28, 29, 2, 3]])
    pixels = torch.randn(4, 24, dtype=torch.bfloat16)
    grid = torch.tensor([[1, 2, 2]])
    with Engine.create(
        tiny_model, tmp_path / "full", ids, max_new_tokens=4, pixel_values=pixels, image_grid_thw=grid
    ) as engine:
        expected = engine.generate()
    with Engine.create(
        tiny_model,
        tmp_path / "partial",
        ids,
        max_new_tokens=4,
        pixel_values=pixels,
        image_grid_thw=grid,
        stop_after_tokens=2,
    ) as engine:
        assert engine.generate() == expected[:2]
    with Engine.resume(tmp_path / "partial") as engine:
        assert engine.generate() == expected


def test_prefill_resume(tiny_model, tmp_path):
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with Engine.create(tiny_model, tmp_path / "full", ids, max_new_tokens=4, chunk_size=2) as engine:
        expected = engine.generate()
    with Engine.create(tiny_model, tmp_path / "partial", ids, max_new_tokens=4, chunk_size=2) as engine:
        original = engine.state.commit

        def interrupt():
            original()
            raise InterruptedError("Simulated power interruption after a committed prefill step")

        engine.state.commit = interrupt
        with pytest.raises(InterruptedError):
            engine.generate()
    with Engine.resume(tmp_path / "partial") as engine:
        assert engine.generate() == expected


def test_sampled_resume(tiny_model, tmp_path):
    from rover.runtime.config import SamplingParams

    ids = torch.tensor([[1, 2, 3, 4]])
    sampling = SamplingParams(do_sample=True, temperature=0.8, top_k=12, top_p=0.9, seed=123)
    kwargs = dict(max_new_tokens=8, runtime_options={"sampling": sampling})
    with Engine.create(tiny_model, tmp_path / "full", ids, **kwargs) as engine:
        expected = engine.generate()
    with Engine.create(tiny_model, tmp_path / "partial", ids, stop_after_tokens=3, **kwargs) as engine:
        assert engine.generate() == expected[:3]
    with Engine.resume(tmp_path / "partial") as engine:
        assert engine.generate() == expected


def test_kivi_unquantized_prefix_matches_dense(tiny_model, tmp_path):
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with Engine.create(tiny_model, tmp_path / "dense", ids, max_new_tokens=4, chunk_size=5) as engine:
        expected = engine.generate()
    with Engine.create(
        tiny_model,
        tmp_path / "kivi",
        ids,
        max_new_tokens=4,
        chunk_size=5,
        cache_format="kivi",
        runtime_options={"kivi_group_size": 16, "kivi_residual_length": 32},
    ) as engine:
        assert engine.generate() == expected
