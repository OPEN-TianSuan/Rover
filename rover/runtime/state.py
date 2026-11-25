"""Durable inference state and checkpoint recovery."""

import gc
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path

import torch

from rover.cache.base import BaseKVCache, build_kv_cache, compute_kv_file_sizes, resolve_kv_cache_metadata
from rover.diagnostics.tracing import tracer
from rover.models.qwen3.config import Qwen3Config
from rover.nn.rotary import RotaryEmbedding
from rover.runtime.config import RuntimeConfig
from rover.runtime.types import ChunkLayerIndex, Phase, PrefillInputs, Range
from rover.storage.files import (
    CacheFile,
    _fsync_dir,
    _load_tensor_checkpoint,
    _save_tensor_checkpoint,
    create_cache_file,
    swap_cache_file,
)


def _initial_cache_format(runtime_config: RuntimeConfig) -> str:
    return "dense" if runtime_config.kv_cache_format == "turboquant" else runtime_config.kv_cache_format


def _max_cache_len(runtime_config: RuntimeConfig, prompt_len: int) -> int:
    return max(runtime_config.max_cache_len, prompt_len + runtime_config.max_decode_tokens)


class GenerationState:
    CONFIG_FILE = "config.json"
    RUNTIME_CONFIG_FILE = "runtime_config.json"
    METADATA_FILE = "metadata.json"
    STATIC_STATE_FILE = "state_static.json"
    DYNAMIC_STATE_FILE = "state_dynamic.json"
    K_FILE = "k.bin"
    V_FILE = "v.bin"
    KV_FILE = "kv.bin"
    H_FILE = "h.bin"
    PENDING_TURBOQUANT_K_FILE = "pending_turboquant_k.bin"
    PENDING_TURBOQUANT_V_FILE = "pending_turboquant_v.bin"
    PENDING_TURBOQUANT_META_FILE = "pending_turboquant.json"
    PIXEL_VALUES_FILE = "pixel_values.bin"
    PIXEL_VALUES_VIDEOS_FILE = "pixel_values_videos.bin"
    PREFILL_INPUTS_FILE = "prefill_inputs.bin"
    VISION_FILE_TEMPLATE = "vision_input_{input_idx}.bin"

    @staticmethod
    def _uses_full_hidden_checkpoint(runtime_config: RuntimeConfig) -> bool:
        return runtime_config.prefill_order in {"layer_first", "scheduled"}

    @staticmethod
    def _keeps_full_hidden_in_memory(runtime_config: RuntimeConfig) -> bool:
        return runtime_config.prefill_order == "layer_first"

    @staticmethod
    def _prefill_hidden_file_nbytes(
        config: Qwen3Config,
        runtime_config: RuntimeConfig,
        prompt_len: int,
    ) -> int:
        elem_size = torch.empty(0, dtype=torch.bfloat16).element_size()
        if GenerationState._uses_full_hidden_checkpoint(runtime_config):
            tc = config.text_config
            return tc.num_hidden_layers * prompt_len * tc.hidden_size * elem_size
        return config.text_config.hidden_size * elem_size

    def __init__(
        self,
        *,
        state_dir: str,
        config: Qwen3Config,
        runtime_config: RuntimeConfig,
        layer_progress: list[int],
        input_ids: torch.Tensor,
        total_chunks: int,
        generated_tokens: Sequence[int],
        decode_kv_tokens: int,
        phase: Phase = Phase.PREPARE_PREFILL,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        cache_format: str,
        cache_metadata: dict | None = None,
        pending_turboquant: dict | None = None,
    ):
        device = runtime_config.device
        prompt_len = input_ids.shape[1]
        max_len = _max_cache_len(runtime_config, prompt_len)

        self._base_dir = Path(state_dir)
        self._config = config
        self._runtime_config = runtime_config
        self._layer_progress = layer_progress
        self._saved_progress = list(layer_progress)
        self._input_ids: torch.Tensor = input_ids.to(device=device, dtype=torch.long)
        self._prefill_inputs_meta: list[dict] | None = None
        self._vision_input_meta: list[dict] = []
        self._static_state_dirty = True
        self.decode_rope_offset: int = prompt_len
        self.pixel_values: torch.Tensor | None = pixel_values
        self.image_grid_thw: torch.Tensor | None = image_grid_thw
        self.pixel_values_videos: torch.Tensor | None = pixel_values_videos
        self.video_grid_thw: torch.Tensor | None = video_grid_thw
        self._pixel_values_meta = None
        self._pixel_values_videos_meta = None
        self._vision_layer_progress: list[int] = []
        self.total_chunks = total_chunks
        self.phase = phase
        self._generated_tokens: list[int] = list(generated_tokens)
        self._decode_kv_tokens = decode_kv_tokens
        self._saved_decode_kv_tokens = decode_kv_tokens
        self._pending_turboquant = pending_turboquant

        tc = config.text_config
        self.num_layers = tc.num_hidden_layers
        self.kv_heads = tc.num_key_value_heads
        self.head_dim = tc.head_dim
        self.hidden_size = tc.hidden_size
        self.max_len = max_len
        self._uses_full_h = self._uses_full_hidden_checkpoint(runtime_config)
        self._prefill_output_hidden: torch.Tensor | None = None
        self._saved_prefill_output_hidden = False
        self._transient_h: dict[tuple[int, int, int], torch.Tensor] = {}

        self._cache_format = cache_format
        self._cache_metadata = cache_metadata
        self._kv_cache: BaseKVCache | None = None
        if phase in {Phase.PREPARE_PREFILL, Phase.PREFILLING} and self._keeps_full_hidden_in_memory(runtime_config):
            self.h = torch.zeros(
                (self.num_layers, 1, prompt_len, self.hidden_size),
                dtype=torch.bfloat16,
            )
        else:
            self.h = None

        self._k_file = CacheFile(self._base_dir / self.K_FILE)
        self._v_file = CacheFile(self._base_dir / self.V_FILE)
        kv_path = self._base_dir / self.KV_FILE
        self._kv_file = CacheFile(kv_path) if kv_path.exists() else None
        self._h_file = CacheFile(self._base_dir / self.H_FILE)

    @classmethod
    def create(
        cls,
        state_dir: str,
        config: Qwen3Config,
        runtime_config: RuntimeConfig,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> "GenerationState":
        base = Path(state_dir)
        tc = config.text_config
        num_layers = tc.num_hidden_layers
        prompt_len = input_ids.shape[1]
        max_len = _max_cache_len(runtime_config, prompt_len)
        h_nbytes = cls._prefill_hidden_file_nbytes(config, runtime_config, prompt_len)
        cache_format = _initial_cache_format(runtime_config)
        key_nbytes, value_nbytes = compute_kv_file_sizes(
            config=config,
            runtime_config=runtime_config,
            max_len=max_len,
            cache_format=cache_format,
        )
        create_cache_file(base / cls.K_FILE, key_nbytes)
        create_cache_file(base / cls.V_FILE, value_nbytes)
        create_cache_file(base / cls.H_FILE, h_nbytes)
        total_chunks = (input_ids.shape[1] + runtime_config.prefill_chunk_size - 1) // runtime_config.prefill_chunk_size
        state = cls(
            state_dir=state_dir,
            config=config,
            runtime_config=runtime_config,
            layer_progress=[0] * num_layers,
            input_ids=input_ids,
            total_chunks=total_chunks,
            generated_tokens=(),
            decode_kv_tokens=0,
            cache_format=cache_format,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        if pixel_values is not None:
            state._pixel_values_meta = _save_tensor_checkpoint(base / cls.PIXEL_VALUES_FILE, pixel_values)
        if pixel_values_videos is not None:
            state._pixel_values_videos_meta = _save_tensor_checkpoint(
                base / cls.PIXEL_VALUES_VIDEOS_FILE, pixel_values_videos
            )
        return state

    @classmethod
    def recover(cls, state_dir: str) -> "GenerationState":
        base = Path(state_dir)
        with open(base / cls.CONFIG_FILE) as f:
            config = Qwen3Config.from_dict(json.load(f))
        with open(base / cls.RUNTIME_CONFIG_FILE) as f:
            runtime_config = RuntimeConfig.from_dict(json.load(f))
        with open(base / cls.METADATA_FILE) as f:
            metadata = json.load(f)
        with open(base / metadata["static_state_file"]) as f:
            static_payload = json.load(f)
        with open(base / metadata["dynamic_state_file"]) as f:
            dynamic_payload = json.load(f)
        input_ids = torch.tensor(static_payload["input_ids"], dtype=torch.long)
        layer_progress = dynamic_payload["layer_progress"]
        total_chunks = metadata["total_chunks"]
        phase = Phase(metadata["phase"])
        generated_tokens = dynamic_payload["generated_tokens"]
        decode_kv_tokens = int(metadata["decode_kv_tokens"])
        image_grid_thw_list = static_payload["image_grid_thw"]
        image_grid_thw = torch.tensor(image_grid_thw_list) if image_grid_thw_list is not None else None
        video_grid_thw_list = static_payload["video_grid_thw"]
        video_grid_thw = torch.tensor(video_grid_thw_list) if video_grid_thw_list is not None else None
        pixel_values_meta = static_payload["pixel_values_meta"]
        pixel_values_videos_meta = static_payload["pixel_values_videos_meta"]
        needs_visual_inputs = phase == Phase.PREPARE_PREFILL
        pixel_values = (
            _load_tensor_checkpoint(base, pixel_values_meta)
            if needs_visual_inputs and pixel_values_meta is not None
            else None
        )
        pixel_values_videos = (
            _load_tensor_checkpoint(base, pixel_values_videos_meta)
            if needs_visual_inputs and pixel_values_videos_meta is not None
            else None
        )
        cache_info = metadata["kv_cache"]
        cache_data = resolve_kv_cache_metadata(
            cache_format=cache_info["format"],
            base_dir=base,
            cache_metadata=cache_info["data"],
        )
        prefill_inputs_meta = static_payload["prefill_inputs_meta"]
        vision_input_meta = static_payload["vision_input_meta"]
        vision_layer_progress = dynamic_payload["vision_layer_progress"]
        assert len(vision_input_meta) == len(vision_layer_progress)
        state = cls(
            state_dir=state_dir,
            config=config,
            runtime_config=runtime_config,
            layer_progress=layer_progress,
            input_ids=input_ids,
            total_chunks=total_chunks,
            generated_tokens=generated_tokens,
            decode_kv_tokens=decode_kv_tokens,
            phase=phase,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            cache_format=cache_info["format"],
            cache_metadata=cache_data,
            pending_turboquant=metadata["pending_turboquant"],
        )
        state._prefill_inputs_meta = prefill_inputs_meta
        state._vision_input_meta = vision_input_meta
        state._vision_layer_progress = vision_layer_progress
        state._pixel_values_meta = pixel_values_meta
        state._pixel_values_videos_meta = pixel_values_videos_meta
        state._static_state_dirty = False
        state.decode_rope_offset = metadata["decode_rope_offset"]
        if phase != Phase.PREPARE_PREFILL:
            state.load_kvh()
        state._saved_prefill_output_hidden = (
            not state._uses_full_h and phase == Phase.PREFILLING and state.completed_chunks == state.total_chunks
        )
        return state

    def begin_prepare_decoding(self, first_token_id: int):
        assert self.phase == Phase.PREFILLING
        self.phase = Phase.PREPARE_DECODING
        self._generated_tokens = [first_token_id]
        self._decode_kv_tokens = 0
        self._saved_decode_kv_tokens = 0
        self.save_metadata()

    def pending_turboquant_paths(self) -> tuple[Path, Path, Path]:
        info = self.pending_turboquant_info
        if info is None:
            return (
                self._base_dir / self.PENDING_TURBOQUANT_K_FILE,
                self._base_dir / self.PENDING_TURBOQUANT_V_FILE,
                self._base_dir / self.PENDING_TURBOQUANT_META_FILE,
            )
        return (
            self._base_dir / info["k_file"],
            self._base_dir / info["v_file"],
            self._base_dir / info["metadata_file"],
        )

    def prefill_cache_paths(self) -> tuple[Path, Path]:
        return self._base_dir / self.K_FILE, self._base_dir / self.V_FILE

    def save_pending_turboquant_progress(self, layers: list[dict]) -> None:
        _, _, meta_path = self.pending_turboquant_paths()
        with open(meta_path, "w") as f:
            json.dump({"completed_layers": len(layers), "layers": layers}, f)

    def init_pending_turboquant(self) -> None:
        key_nbytes, value_nbytes = compute_kv_file_sizes(
            config=self._config,
            runtime_config=self._runtime_config,
            max_len=self.max_len,
            cache_format="turboquant",
        )
        k_path, v_path, meta_path = self.pending_turboquant_paths()
        create_cache_file(k_path, key_nbytes)
        create_cache_file(v_path, value_nbytes)
        self._pending_turboquant = {
            "enabled": True,
            "k_file": k_path.name,
            "v_file": v_path.name,
            "metadata_file": meta_path.name,
        }
        self.save_pending_turboquant_progress([])
        self.save_metadata()

    def clear_pending_turboquant(self) -> None:
        self._pending_turboquant = None
        for path in self.pending_turboquant_paths():
            if path.exists():
                path.unlink()

    def load_pending_turboquant_progress(self) -> dict:
        _, _, meta_path = self.pending_turboquant_paths()
        if not meta_path.exists():
            return {"completed_layers": 0, "layers": []}
        with open(meta_path) as f:
            return json.load(f)

    def _rewrite_prefill_kv_files(self) -> None:
        key_nbytes = self.kv_cache.prefill_key_file_size()
        value_nbytes = self.kv_cache.prefill_value_file_size()
        k_tmp_path, k_tmp_file = swap_cache_file(self._base_dir / self.K_FILE, key_nbytes)
        v_tmp_path, v_tmp_file = swap_cache_file(self._base_dir / self.V_FILE, value_nbytes)
        try:
            for layer in range(self.num_layers):
                self.kv_cache.save_prefill_layer(layer, 0, self.prompt_len, k_tmp_file, v_tmp_file)
        finally:
            k_tmp_file.close()
            v_tmp_file.close()
        self._k_file.close()
        self._v_file.close()
        os.replace(k_tmp_path, self._base_dir / self.K_FILE)
        os.replace(v_tmp_path, self._base_dir / self.V_FILE)
        _fsync_dir(self._base_dir)
        self._k_file = CacheFile(self._base_dir / self.K_FILE)
        self._v_file = CacheFile(self._base_dir / self.V_FILE)

    def _open_decode_file(self, nbytes: int) -> None:
        path = self._base_dir / self.KV_FILE
        create_cache_file(path, nbytes)
        if self._kv_file is not None:
            self._kv_file.close()
        self._kv_file = CacheFile(path)

    def _decode_file_nbytes(self) -> int:
        return max(self._runtime_config.max_decode_tokens - 1, 0) * self.kv_cache.decode_record_nbytes()

    def _release_prefill_hidden(self) -> None:
        self.h = None
        self._prefill_output_hidden = None
        self._transient_h.clear()
        gc.collect()

    def _convert_dense_prefix_to_turboquant(self) -> None:
        from rover.cache.dense import DenseKVCache

        assert isinstance(self.kv_cache, DenseKVCache)
        elem_size = torch.empty(0, dtype=torch.bfloat16).element_size()
        token_stride = self.kv_heads * self.head_dim * elem_size
        prefix_nbytes = self.prompt_len * token_stride
        self.drop_kv_cache()
        gc.collect()
        self.set_kv_cache(
            build_kv_cache(
                config=self._config,
                runtime_config=self._runtime_config,
                max_len=self.max_len,
                cache_format="turboquant",
            )
        )
        gc.collect()

        device = torch.device(self._runtime_config.device)
        with torch.device(device):
            key_states = torch.empty((1, self.kv_heads, self.prompt_len, self.head_dim), dtype=torch.bfloat16)
            value_states = torch.empty_like(key_states)
        for layer in range(self.num_layers):
            offset = (layer * self.max_len) * token_stride
            self._k_file.read_tensor(key_states[0].permute(1, 0, 2), prefix_nbytes, offset)
            self._v_file.read_tensor(value_states[0].permute(1, 0, 2), prefix_nbytes, offset)
            self.kv_cache.update_kv(Range(layer, 0, self.prompt_len), key_states, value_states)

    def prepare_decoding(self):
        assert self.phase == Phase.PREPARE_DECODING
        if self._runtime_config.turboquant_overlap_decode:
            assert self.kv_cache.cache_format() == "dense"
            self._release_prefill_hidden()
            if self._pending_turboquant is None:
                self.init_pending_turboquant()
            self.kv_cache.begin_decode(self.prompt_len)
            self._open_decode_file(self._decode_file_nbytes())
            self.phase = Phase.DECODING_DENSE_OVERLAP
            self.save_metadata()
            return
        if self.kv_cache.cache_format() == "dense" and self._runtime_config.kv_cache_format == "turboquant":
            self._release_prefill_hidden()
            self._convert_dense_prefix_to_turboquant()
            self._rewrite_prefill_kv_files()
        else:
            self._release_prefill_hidden()
        self.kv_cache.begin_decode(self.prompt_len)
        self._open_decode_file(self._decode_file_nbytes())
        self.phase = Phase.DECODING
        self.save_metadata()

    def rewrite_decode_file(self) -> None:
        assert self._decode_kv_tokens >= 0
        path = self._base_dir / self.KV_FILE
        tmp_path, tmp_file = swap_cache_file(path, self._decode_file_nbytes())
        try:
            if self._decode_kv_tokens > 0:
                self.kv_cache.save_decode_tokens(0, self._decode_kv_tokens, tmp_file)
        finally:
            tmp_file.close()
        if self._kv_file is not None:
            self._kv_file.close()
        os.replace(tmp_path, path)
        _fsync_dir(self._base_dir)
        self._kv_file = CacheFile(path)
        self._saved_decode_kv_tokens = self._decode_kv_tokens

    def switch_to_turboquant(self, kv_cache: BaseKVCache) -> None:
        assert kv_cache.cache_format() == "turboquant"
        pending_k_path, pending_v_path, _ = self.pending_turboquant_paths()
        self.set_kv_cache(kv_cache)
        self.kv_cache.begin_decode(self.prompt_len)
        self._k_file.close()
        self._v_file.close()
        os.replace(pending_k_path, self._base_dir / self.K_FILE)
        os.replace(pending_v_path, self._base_dir / self.V_FILE)
        self._k_file = CacheFile(self._base_dir / self.K_FILE)
        self._v_file = CacheFile(self._base_dir / self.V_FILE)
        self.rewrite_decode_file()
        _, _, meta_path = self.pending_turboquant_paths()
        if meta_path.exists():
            meta_path.unlink()
        self._pending_turboquant = None
        self.phase = Phase.DECODING
        self.save_metadata()

    def close(self):
        self._k_file.close()
        self._v_file.close()
        if self._kv_file is not None:
            self._kv_file.close()
        self._h_file.close()

    @property
    def kv_cache(self) -> BaseKVCache:
        if self._kv_cache is None:
            self._kv_cache = build_kv_cache(
                config=self._config,
                runtime_config=self._runtime_config,
                max_len=self.max_len,
                cache_format=self._cache_format,
                cache_metadata=self._cache_metadata,
            )
        return self._kv_cache

    def drop_kv_cache(self) -> None:
        self._kv_cache = None

    def set_kv_cache(self, cache: BaseKVCache) -> None:
        self._kv_cache = cache
        self._cache_format = cache.cache_format()
        self._cache_metadata = cache.get_metadata()

    def _kv_cache_metadata_payload(self) -> dict:
        if self._kv_cache is None:
            return self._cache_metadata or {}
        self._cache_metadata = self._kv_cache.get_metadata()
        return self._kv_cache.metadata_payload(self._base_dir)

    def save_metadata(self):
        config_path = self._base_dir / self.CONFIG_FILE
        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump(self._config.to_dict(), f)
        runtime_path = self._base_dir / self.RUNTIME_CONFIG_FILE
        if not runtime_path.exists():
            with open(runtime_path, "w") as f:
                json.dump(self._runtime_config.to_dict(), f)
        static_path = self._base_dir / self.STATIC_STATE_FILE
        if self._static_state_dirty or not static_path.exists():
            with open(static_path, "w") as f:
                json.dump(
                    {
                        "input_ids": self._input_ids.tolist(),
                        "image_grid_thw": self.image_grid_thw.tolist() if self.image_grid_thw is not None else None,
                        "video_grid_thw": self.video_grid_thw.tolist() if self.video_grid_thw is not None else None,
                        "pixel_values_meta": self._pixel_values_meta,
                        "pixel_values_videos_meta": self._pixel_values_videos_meta,
                        "prefill_inputs_meta": self._prefill_inputs_meta,
                        "vision_input_meta": self._vision_input_meta,
                    },
                    f,
                )
            self._static_state_dirty = False
        with open(self._base_dir / self.DYNAMIC_STATE_FILE, "w") as f:
            json.dump(
                {
                    "layer_progress": self._layer_progress,
                    "vision_layer_progress": self._vision_layer_progress,
                    "generated_tokens": self._generated_tokens,
                },
                f,
            )
        with open(self._base_dir / self.METADATA_FILE, "w") as f:
            metadata = {
                "total_chunks": self.total_chunks,
                "phase": self.phase.value,
                "decode_kv_tokens": self._decode_kv_tokens,
                "static_state_file": self.STATIC_STATE_FILE,
                "dynamic_state_file": self.DYNAMIC_STATE_FILE,
                "kv_cache": {"format": self._cache_format, "data": self._kv_cache_metadata_payload()},
                "decode_rope_offset": self.decode_rope_offset,
                "pending_turboquant": self._pending_turboquant,
            }
            json.dump(metadata, f)

    # ---- KV/H IO ----

    def commit_prefill_inputs(self, prefill_inputs: PrefillInputs) -> None:
        assert self.phase == Phase.PREPARE_PREFILL
        tensors = [
            prefill_inputs.inputs_embeds,
            prefill_inputs.pos_emb._cos,
            prefill_inputs.pos_emb._sin,
            prefill_inputs.visual_pos_masks,
            *prefill_inputs.deepstack_visual_embeds,
        ]

        path = self._base_dir / self.PREFILL_INPUTS_FILE
        total_bytes = sum(t.nbytes for t in tensors)
        create_cache_file(path, total_bytes)

        meta = []
        offset = 0
        with CacheFile(path) as f:
            for t in tensors:
                tc = t.contiguous().cpu()
                f.write_tensor(tc, offset)
                meta.append({"shape": list(tc.shape), "dtype": str(tc.dtype).replace("torch.", ""), "offset": offset})
                offset += tc.nbytes
            f.sync()
            f.discard_all_page_cache()
        self._prefill_inputs_meta = meta
        self._static_state_dirty = True
        self.decode_rope_offset = prefill_inputs.decode_rope_offset
        self.phase = Phase.PREFILLING
        self.save_metadata()

    def load_prefill_inputs(self) -> PrefillInputs:
        meta = self._prefill_inputs_meta
        assert meta is not None
        path = self._base_dir / self.PREFILL_INPUTS_FILE
        device = self._runtime_config.device

        def read(f: CacheFile, m: dict) -> torch.Tensor:
            dtype = getattr(torch, m["dtype"])
            t = torch.empty(m["shape"], dtype=dtype)
            f.read_tensor(t, t.nbytes, m["offset"])
            return t.to(device)

        with CacheFile(path) as f:
            inputs_embeds = read(f, meta[0])
            pos_emb_cos = read(f, meta[1])
            pos_emb_sin = read(f, meta[2])
            visual_pos_masks = read(f, meta[3])
            deepstack_visual_embeds = [read(f, m) for m in meta[4:]]
        return PrefillInputs(
            inputs_embeds,
            RotaryEmbedding(pos_emb_cos, pos_emb_sin),
            visual_pos_masks,
            deepstack_visual_embeds,
            self.decode_rope_offset,
        )

    def _vision_path(self, input_idx: int) -> Path:
        return self._base_dir / self.VISION_FILE_TEMPLATE.format(input_idx=input_idx)

    def _vision_slot_nbytes(self, meta: dict) -> int:
        dtype = getattr(torch, meta["dtype"])
        numel = math.prod(meta["shape"])
        return int(torch.empty(0, dtype=dtype).element_size() * numel)

    def init_vision_checkpoint(self, input_idx: int, stem_hidden: torch.Tensor, num_layers: int) -> None:
        assert input_idx == len(self._vision_input_meta)
        hidden_cpu = stem_hidden.contiguous().cpu()
        meta = {
            "shape": list(hidden_cpu.shape),
            "dtype": str(hidden_cpu.dtype).replace("torch.", ""),
        }
        create_cache_file(self._vision_path(input_idx), (num_layers + 1) * hidden_cpu.nbytes)
        with CacheFile(self._vision_path(input_idx)) as f:
            f.write_tensor(hidden_cpu, 0)
            f.sync()
            f.discard_all_page_cache()
        self._vision_input_meta.append(meta)
        self._vision_layer_progress.append(0)
        self._static_state_dirty = True
        self.save_metadata()

    def save_vision_layer(self, input_idx: int, layer_idx: int, hidden_states: torch.Tensor) -> None:
        meta = self._vision_input_meta[input_idx]
        assert self._vision_layer_progress[input_idx] == layer_idx
        slot_nbytes = self._vision_slot_nbytes(meta)
        offset = (layer_idx + 1) * slot_nbytes
        with CacheFile(self._vision_path(input_idx)) as f:
            f.write_tensor(hidden_states.contiguous().cpu(), offset)
            f.sync()
            f.discard_all_page_cache()
        self._vision_layer_progress[input_idx] = layer_idx + 1
        self.save_metadata()

    def load_vision_stage(
        self,
        input_idx: int,
        stage_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        meta = self._vision_input_meta[input_idx]
        dtype = getattr(torch, meta["dtype"])
        hidden = torch.empty(meta["shape"], dtype=dtype)
        slot_nbytes = self._vision_slot_nbytes(meta)
        with CacheFile(self._vision_path(input_idx)) as f:
            f.read_tensor(hidden, slot_nbytes, stage_idx * slot_nbytes)
        return hidden.to(device)

    def vision_layer_progress(self, input_idx: int) -> int:
        return self._vision_layer_progress[input_idx]

    @property
    def vision_checkpoint_count(self) -> int:
        return len(self._vision_input_meta)

    def _h_offset(self, layer: int, start: int) -> int:
        stride = self.hidden_size * torch.empty(0, dtype=torch.bfloat16).element_size()
        return (layer * self.prompt_len + start) * stride

    def write_h(self, r: Range):
        offset = self._h_offset(r.layer, r.start)
        if self.h is not None:
            h_states = self.h[r.layer, 0, r.start : r.end, :]
        else:
            h_states = self._transient_h[(r.layer, r.start, r.end)][0]
        self._h_file.write_tensor(h_states, offset)

    def read_h(self, r: Range):
        nbytes = r.length * self.hidden_size * torch.empty(0, dtype=torch.bfloat16).element_size()
        offset = self._h_offset(r.layer, r.start)
        if self.h is not None:
            self._h_file.read_tensor(self.h[r.layer, 0, r.start : r.end, :], nbytes, offset)
            return
        h_states = torch.empty((1, r.length, self.hidden_size), dtype=torch.bfloat16)
        self._h_file.read_tensor(h_states[0], nbytes, offset)
        self._transient_h[(r.layer, r.start, r.end)] = h_states

    def save_prefill_kv(self, idx: ChunkLayerIndex) -> None:
        self.kv_cache.save_prefill_layer(idx.layer, idx.start, idx.end, self._k_file, self._v_file)
        self._k_file.sync()
        self._v_file.sync()
        self._k_file.discard_all_page_cache()
        self._v_file.discard_all_page_cache()

    def load_prefill_kv(self, idx: ChunkLayerIndex) -> None:
        self.kv_cache.load_prefill_layer(idx.layer, idx.start, idx.end, self._k_file, self._v_file)

    def save_prefill_h(self, idx: ChunkLayerIndex) -> None:
        self.write_h(Range(idx.layer, idx.start, idx.end))
        self._h_file.sync()
        self._h_file.discard_all_page_cache()

    def mark_prefill_cell_persisted(self, idx: ChunkLayerIndex) -> None:
        assert self.phase == Phase.PREFILLING
        assert self._layer_progress[idx.layer] == idx.chunk_id
        self._layer_progress[idx.layer] = idx.chunk_id + 1
        self._saved_progress[idx.layer] = idx.chunk_id + 1
        self.save_metadata()

    def write_prefill_output_hidden(self) -> None:
        assert self._prefill_output_hidden is not None
        self._h_file.write_tensor(self._prefill_output_hidden, 0)

    def read_prefill_output_hidden(self) -> None:
        self._prefill_output_hidden = torch.empty((1, 1, self.hidden_size), dtype=torch.bfloat16)
        self._h_file.read_tensor(self._prefill_output_hidden, self._prefill_output_hidden.nbytes, 0)

    def load_kvh(self):
        chunk_size = self._runtime_config.prefill_chunk_size
        for layer in range(self.num_layers):
            end = self.prompt_len
            if self.phase in {Phase.PREPARE_PREFILL, Phase.PREFILLING}:
                chunks_done = self._layer_progress[layer]
                end = chunks_done * chunk_size if chunks_done < self.total_chunks else self.prompt_len
            self.kv_cache.load_prefill_layer(layer, 0, end, self._k_file, self._v_file)
            if end > 0 and self.h is not None:
                self.read_h(Range(layer, 0, end))
        if not self._uses_full_h and self.phase == Phase.PREFILLING and self.completed_chunks == self.total_chunks:
            self.read_prefill_output_hidden()
        if self.phase in {Phase.DECODING_DENSE_OVERLAP, Phase.DECODING} and self._decode_kv_tokens > 0:
            assert self._kv_file is not None
            self.kv_cache.load_decode_tokens(self._kv_file, self._decode_kv_tokens)

    @property
    def config(self) -> Qwen3Config:
        return self._config

    @property
    def runtime_config(self) -> RuntimeConfig:
        return self._runtime_config

    @property
    def layer_progress(self) -> list[int]:
        return self._layer_progress

    @property
    def input_ids(self) -> torch.Tensor:
        return self._input_ids

    @property
    def generated_tokens(self) -> list[int]:
        return self._generated_tokens

    @property
    def pending_turboquant_info(self) -> dict | None:
        return self._pending_turboquant

    @property
    def cache_format(self) -> str:
        return self._cache_format

    @property
    def decode_kv_tokens(self) -> int:
        return self._decode_kv_tokens

    @property
    def prompt_len(self) -> int:
        return self._input_ids.shape[1]

    @property
    def completed_layers(self) -> int:
        for idx, progress in enumerate(self._layer_progress):
            if progress < self.total_chunks:
                return idx
        return self.num_layers

    @property
    def completed_chunks(self) -> int:
        return min(self._layer_progress)

    def get_kv_for_attention(self, layer: int):
        return self.kv_cache.get_kv_for_attention(layer)

    def update_kv(
        self,
        r: Range,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        assert key_states.shape == (1, self.kv_heads, r.length, self.head_dim)
        assert value_states.shape == (1, self.kv_heads, r.length, self.head_dim)
        self.kv_cache.update_kv(r, key_states, value_states)

    def get_h(self, r: Range) -> torch.Tensor:
        if self.h is not None:
            return self.h[r.layer, :, r.start : r.end, :].to(device=self._runtime_config.device)
        key = (r.layer, r.start, r.end)
        assert key in self._transient_h
        return self._transient_h[key].to(device=self._runtime_config.device)

    def drop_h(self, r: Range) -> None:
        self._transient_h.pop((r.layer, r.start, r.end), None)

    def update_h(self, r: Range, h_states: torch.Tensor):
        assert h_states.shape == (1, r.length, self.hidden_size)
        if self.h is not None:
            self.h[r.layer, :, r.start : r.end, :].copy_(h_states)
            return
        self._transient_h[(r.layer, r.start, r.end)] = h_states.detach().cpu()
        if r.layer == self.num_layers - 1 and r.end == self.prompt_len:
            self.update_prefill_output_hidden(h_states[:, -1:, :])

    def get_prefill_output_hidden(self) -> torch.Tensor:
        assert self._prefill_output_hidden is not None
        return self._prefill_output_hidden.to(device=self._runtime_config.device)

    def update_prefill_output_hidden(self, hidden_states: torch.Tensor) -> None:
        assert hidden_states.shape == (1, 1, self.hidden_size)
        self._prefill_output_hidden = hidden_states.detach().cpu()
        self._saved_prefill_output_hidden = False

    def advance_prefill(self, idx: ChunkLayerIndex) -> None:
        self._layer_progress[idx.layer] = idx.chunk_id + 1

    def advance_decode(self, next_token_id: int) -> None:
        self._generated_tokens.append(next_token_id)
        self._decode_kv_tokens += 1

    def commit(self):
        dirty_files: list[CacheFile] = []
        if self.phase == Phase.PREFILLING:
            chunk_size = self._runtime_config.prefill_chunk_size
            for layer, (saved, current) in enumerate(zip(self._saved_progress, self._layer_progress)):
                if current > saved:
                    start = saved * chunk_size
                    end = min(current * chunk_size, self.prompt_len)
                    self.kv_cache.save_prefill_layer(layer, start, end, self._k_file, self._v_file)
                    if self.h is not None:
                        self.write_h(Range(layer, start, end))
                    self._saved_progress[layer] = current
                    dirty_files.extend((self._k_file, self._v_file))
                    if self.h is not None:
                        dirty_files.append(self._h_file)
            if (
                not self._uses_full_h
                and self.completed_chunks == self.total_chunks
                and self._prefill_output_hidden is not None
                and not self._saved_prefill_output_hidden
            ):
                self.write_prefill_output_hidden()
                self._saved_prefill_output_hidden = True
                dirty_files.append(self._h_file)
        elif self.phase in {Phase.DECODING_DENSE_OVERLAP, Phase.DECODING}:
            if self._decode_kv_tokens > self._saved_decode_kv_tokens:
                assert self._kv_file is not None
                with tracer.timer("commit_save_kv", sync_cuda=self._runtime_config.sync_cuda_timers):
                    self.kv_cache.save_decode_tokens(
                        self._saved_decode_kv_tokens,
                        self._decode_kv_tokens,
                        self._kv_file,
                    )
                self._saved_decode_kv_tokens = self._decode_kv_tokens
                dirty_files.append(self._kv_file)
        with tracer.timer("commit_save_metadata", sync_cuda=self._runtime_config.sync_cuda_timers):
            self.save_metadata()
        for file in {id(file): file for file in dirty_files}.values():
            file.sync()
            file.discard_all_page_cache()
