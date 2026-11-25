"""Public lifecycle API for local models and durable inference sessions."""

from dataclasses import replace
from pathlib import Path

import torch

from rover.models.qwen3.loading import load_model_config
from rover.runtime.config import DEFAULT_SAMPLING_PARAMS, RuntimeConfig, SamplingParams
from rover.runtime.runner import GenerationRunner
from rover.runtime.state import GenerationState


class Engine:
    def __init__(self, state: GenerationState):
        self.state = state
        try:
            self.model = GenerationRunner(state).eval()
        except BaseException:
            state.close()
            raise
        self._closed = False

    @classmethod
    def create(
        cls,
        model_dir,
        state_dir,
        input_ids,
        *,
        device="cpu",
        max_new_tokens=128,
        chunk_size=256,
        prefill_order="chunk_first",
        strategy="chunk_first",
        cache_format="dense",
        cuda_graph=False,
        stop_after_tokens=None,
        runtime_options=None,
        **visual_inputs,
    ):
        model_dir = str(Path(model_dir).expanduser().resolve())
        state_dir = Path(state_dir).expanduser().resolve()
        if state_dir.exists() and any(state_dir.iterdir()):
            raise FileExistsError(f"State directory is not empty: {state_dir}. Use resume or a new directory.")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [1, nonzero sequence length]")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable in this environment; use --device cpu or install compatible PyTorch."
            )
        config = load_model_config(model_dir)
        config.validate()
        options = dict(
            model_dir=model_dir,
            device=device,
            state_dir=str(state_dir),
            stop_after_tokens=stop_after_tokens,
            max_decode_tokens=max_new_tokens,
            max_cache_len=int(input_ids.shape[1]) + max_new_tokens,
            max_batch_size=1,
            prefill_chunk_size=chunk_size,
            prefill_order=prefill_order,
            prefill_schedule_strategy=strategy,
            sampling=DEFAULT_SAMPLING_PARAMS,
            kv_cache_format=cache_format,
            cuda_graph=cuda_graph,
        )
        if runtime_options:
            conflicts = set(runtime_options) & (set(options) - {"sampling"})
            if conflicts:
                raise ValueError(f"Use explicit Engine arguments for: {sorted(conflicts)}")
            options.update(runtime_options)
        if isinstance(options["sampling"], dict):
            options["sampling"] = SamplingParams(**options["sampling"])
        runtime = RuntimeConfig(**options)
        state = GenerationState.create(str(state_dir), config, runtime, input_ids, **visual_inputs)
        state.save_metadata()
        return cls(state)

    @classmethod
    def resume(cls, state_dir):
        state = GenerationState.recover(str(Path(state_dir).expanduser().resolve()))
        state._runtime_config = replace(state.runtime_config, stop_after_tokens=None)
        return cls(state)

    def generate(self, on_token=None):
        if self._closed:
            raise RuntimeError("Engine is closed")
        return self.model.generate(on_token=on_token)

    def close(self):
        if not self._closed:
            self.state.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
