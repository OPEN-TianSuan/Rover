"""Generation orchestration, weight residency and cache transitions."""

import gc
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from time import perf_counter
from typing import Any

import torch
from torch import nn

from rover.cache.base import build_kv_cache
from rover.cache.dense import DenseKVCache
from rover.cache.turboquant import (
    TurboQuantGraphBuffers,
    TurboQuantKVCache,
)
from rover.diagnostics.logging import logger
from rover.diagnostics.memory import capture_gpu_mem, reset_peak
from rover.diagnostics.tracing import tracer
from rover.models.quantization import (
    make_visual_marlin_shapes,
    resolve_gptq_text_config,
)
from rover.models.qwen3.loading import is_minicpmv_config, minicpmv_vision_classes
from rover.models.qwen3.multimodal import VisualInput, build_visual_inputs, compute_rope_index
from rover.models.qwen3.position import apply_interleaved_mrope, build_rotary_inv_freq, build_vision_rotary_inv_freq
from rover.models.qwen3.text import Qwen3Model
from rover.models.qwen3.vision import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VisionModel,
)
from rover.nn.modules import InferenceModule, maybe_autocast
from rover.nn.rotary import RotaryEmbedding
from rover.runtime.graph import DecodeGraphRunner
from rover.runtime.sampling import Sampling
from rover.runtime.state import GenerationState
from rover.runtime.types import ChunkLayerIndex, Phase, PrefillInputs, Range
from rover.scheduling.prefill import PrefillScheduler, PrefillScheduleStrategy, build_prefill_strategy
from rover.storage.files import CacheFile
from rover.storage.weights import ModelLoader


class GenerationRunner(InferenceModule):
    """Run a request through vision encoding, prefill, and resumable decoding."""

    def __init__(self, state: GenerationState):
        self.config = state.config
        self.runtime_config = state.runtime_config
        self.hidden_size = state.config.text_config.hidden_size
        self.vocab_size = state.config.text_config.vocab_size
        self.num_layers = self.config.text_config.num_hidden_layers

        self.state = state
        self._device: torch.device = torch.device(self.runtime_config.device)
        self.config._attn_implementation = "eager"
        self.config.text_config._attn_implementation = "eager"
        self.config.vision_config._attn_implementation = "eager"
        self.blank_visual_marlin = os.environ.get("ROVER_BLANK_VISUAL_MARLIN") == "1" and not is_minicpmv_config(
            self.config
        )
        self.text_quant_config = resolve_gptq_text_config(
            getattr(self.config, "quantization_config", None),
            device=self.runtime_config.device,
        )
        if self.blank_visual_marlin:
            make_visual_marlin_shapes(self.config.vision_config)
        visual_quant_config = self.text_quant_config if self.blank_visual_marlin else None

        with tracer.timer("allocate_tensor", print_on_exit=True), torch.device("meta"):
            super().__init__(self.config)
            self.model = Qwen3Model(
                self.config,
                self.runtime_config,
                text_quant_config=self.text_quant_config,
            )
            self.visual = Qwen3VisionModel(
                self.config.vision_config,
                visual_quant_config=visual_quant_config,
            )
            if is_minicpmv_config(self.config):
                siglip_cls, resampler_cls = minicpmv_vision_classes(self.runtime_config.model_dir)
                self.config.vision_config._attn_implementation = "eager"
                self.minicpm_vpm = siglip_cls(self.config.vision_config)
                if self.config.drop_vision_last_layer:
                    self.minicpm_vpm.encoder.layers = self.minicpm_vpm.encoder.layers[:-1]
                self.minicpm_vpm.embed_dim = self.minicpm_vpm.embeddings.embed_dim
                self.minicpm_resampler = resampler_cls(
                    num_queries=self.config.query_num,
                    embed_dim=self.hidden_size,
                    num_heads=self.hidden_size // 128,
                    kv_dim=self.minicpm_vpm.embed_dim,
                    adaptive=True,
                    batch_infer=self.config.batch_3d_resampler,
                )
            self.register_buffer("rotary_inv_freq", build_rotary_inv_freq(self.config.text_config), persistent=False)
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self._materialize_derived_buffers()

        self.sampler = Sampling(self.runtime_config, len(state.generated_tokens))
        self.model_loader: ModelLoader = ModelLoader(self.runtime_config.model_dir, device=self._device)

        self.loaded_layers: set[int] = set()
        self.misc_loaded: bool = False
        self.visual_loaded: bool = False
        self.last_phase_timings_ms: dict[str, float] = {}
        self.last_phase_memory_mb: dict[str, dict[str, float]] = {}
        self._pending_turboquant_executor: ThreadPoolExecutor | None = None
        self._pending_turboquant_future: Future[TurboQuantKVCache] | None = None
        self._pending_turboquant_started_at: float | None = None

    @contextmanager
    def _phase_timer(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            reset_peak()
        started_at = perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (perf_counter() - started_at) * 1000.0
            self.last_phase_timings_ms[name] = self.last_phase_timings_ms.get(name, 0.0) + elapsed_ms
            stats = capture_gpu_mem()
            if stats is not None:
                self.last_phase_memory_mb[name] = stats.to_dict()

    def _add_phase_timing_ms(self, name: str, elapsed_ms: float) -> None:
        self.last_phase_timings_ms[name] = self.last_phase_timings_ms.get(name, 0.0) + elapsed_ms

    def _record_pending_turboquant_prefix_time(self) -> None:
        assert self._pending_turboquant_started_at is not None
        elapsed_ms = (perf_counter() - self._pending_turboquant_started_at) * 1000.0
        self._add_phase_timing_ms("kv_cache_compress_prefix_async", elapsed_ms)
        self._add_phase_timing_ms("kv_cache_compress", elapsed_ms)
        self._pending_turboquant_started_at = None

    def _log_prefill_memory_model(self, prefill_inputs: PrefillInputs) -> None:
        prefill_nbytes = sum(
            t.nbytes
            for t in (
                prefill_inputs.inputs_embeds,
                prefill_inputs.pos_emb._cos,
                prefill_inputs.pos_emb._sin,
                prefill_inputs.visual_pos_masks,
                *prefill_inputs.deepstack_visual_embeds,
            )
        )
        logger.info(
            "[memory-model] prompt_len=%d prefill_inputs=%.2fGiB max_len=%d cache=%s",
            int(prefill_inputs.inputs_embeds.shape[1]),
            prefill_nbytes / 1024**3,
            int(self.state.max_len),
            self.state.cache_format,
        )

    def _build_pending_turboquant_cache(self) -> TurboQuantKVCache:
        progress = self.state.load_pending_turboquant_progress()
        layer_metas = list(progress["layers"])
        pending_cache = build_kv_cache(
            config=self.config,
            runtime_config=self.runtime_config,
            max_len=self.state.max_len,
            cache_format="turboquant",
            cache_metadata={"decode_base_len": None, "layers": layer_metas},
        )
        assert isinstance(pending_cache, TurboQuantKVCache)

        pending_k_path, pending_v_path, _ = self.state.pending_turboquant_paths()
        dense_k_path, dense_v_path = self.state.prefill_cache_paths()
        pending_k_file = CacheFile(pending_k_path)
        pending_v_file = CacheFile(pending_v_path)
        dense_k_file = CacheFile(dense_k_path)
        dense_v_file = CacheFile(dense_v_path)
        try:
            for layer in range(len(layer_metas)):
                pending_cache.load_prefill_layer(layer, 0, self.state.prompt_len, pending_k_file, pending_v_file)

            elem_size = torch.empty(0, dtype=torch.bfloat16).element_size()
            token_stride = self.state.kv_heads * self.state.head_dim * elem_size
            prefix_nbytes = self.state.prompt_len * token_stride
            stream = torch.cuda.Stream(device=self._device) if self._device.type == "cuda" else None
            with torch.device(self._device):
                key_states = torch.empty(
                    (1, self.state.kv_heads, self.state.prompt_len, self.state.head_dim),
                    dtype=torch.bfloat16,
                )
                value_states = torch.empty_like(key_states)
            for layer in range(len(layer_metas), self.num_layers):
                offset = (layer * self.state.max_len) * token_stride
                with torch.cuda.stream(stream) if stream is not None else nullcontext():
                    dense_k_file.read_tensor(key_states[0].permute(1, 0, 2), prefix_nbytes, offset)
                    dense_v_file.read_tensor(value_states[0].permute(1, 0, 2), prefix_nbytes, offset)
                    pending_cache.update_kv(Range(layer, 0, self.state.prompt_len), key_states, value_states)
                if stream is not None:
                    stream.synchronize()
                pending_cache.save_prefill_layer(layer, 0, self.state.prompt_len, pending_k_file, pending_v_file)
                layer_metas.append(pending_cache.layers[layer].get_metadata())
                self.state.save_pending_turboquant_progress(layer_metas)
            return pending_cache
        finally:
            dense_k_file.close()
            dense_v_file.close()
            pending_k_file.close()
            pending_v_file.close()

    def _ensure_pending_turboquant_future(self) -> None:
        if self._pending_turboquant_future is not None:
            return
        self._pending_turboquant_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rover-turboquant")
        self._pending_turboquant_started_at = perf_counter()
        self._pending_turboquant_future = self._pending_turboquant_executor.submit(self._build_pending_turboquant_cache)

    def _switch_to_turboquant(self) -> None:
        assert self._pending_turboquant_future is not None
        pending_cache = self._pending_turboquant_future.result()
        self._record_pending_turboquant_prefix_time()
        dense_cache = self.state.kv_cache
        assert isinstance(dense_cache, DenseKVCache)
        suffix_len = self.state.decode_kv_tokens
        if suffix_len > 0:
            torch.cuda.synchronize()
            started_at = perf_counter()
            for layer in range(self.num_layers):
                key_states, value_states = dense_cache.get_kv(
                    Range(layer, self.state.prompt_len, self.state.prompt_len + suffix_len)
                )
                pending_cache.update_kv(
                    Range(layer, self.state.prompt_len, self.state.prompt_len + suffix_len),
                    key_states,
                    value_states,
                )
            torch.cuda.synchronize()
            elapsed_ms = (perf_counter() - started_at) * 1000.0
            self._add_phase_timing_ms("kv_cache_compress_suffix_sync", elapsed_ms)
            self._add_phase_timing_ms("kv_cache_compress", elapsed_ms)
        self.state.switch_to_turboquant(pending_cache)
        if self._pending_turboquant_executor is not None:
            self._pending_turboquant_executor.shutdown(wait=False)
        self._pending_turboquant_executor = None
        self._pending_turboquant_future = None

    def _capture_decode_graph_runner(self) -> DecodeGraphRunner:
        is_turboquant = isinstance(self.state.kv_cache, TurboQuantKVCache)
        is_dense = isinstance(self.state.kv_cache, DenseKVCache)
        assert is_dense or is_turboquant, (
            f"cuda_graph requires dense or turboquant kv_cache, got {type(self.state.kv_cache)}"
        )

        tc = self.config.text_config
        device = self._device
        logits_dtype = self.lm_head.weight.dtype
        max_len = self.state.max_len
        num_heads = tc.num_attention_heads
        kv_heads = tc.num_key_value_heads
        head_dim = tc.head_dim
        initial_kv_len = self.state.prompt_len + self.state.decode_kv_tokens

        token_buf = torch.zeros((1, 1), dtype=torch.long, device=device)
        position_ids_buf = torch.zeros((3, 1, 1), dtype=torch.long, device=device)
        attn_mask = torch.full((1, 1, 1, max_len), float("-inf"), dtype=torch.float32, device=device)
        attn_mask[:, :, :, :initial_kv_len] = 0.0
        logits_out = torch.zeros((1, tc.vocab_size), dtype=logits_dtype, device=device)

        if is_dense:
            write_idx_exp = torch.zeros((1, kv_heads, 1, head_dim), dtype=torch.long, device=device)
            key_caches = [self.state.kv_cache._k[layer_index] for layer_index in range(self.num_layers)]
            val_caches = [self.state.kv_cache._v[layer_index] for layer_index in range(self.num_layers)]

            def _step():
                hidden = self.model.embed_tokens(token_buf)
                pos_emb = self._rotary_position_embeddings(position_ids_buf)
                for layer_index in range(self.num_layers):
                    hidden = self.model.layers[layer_index]._graph_forward(
                        hidden, pos_emb, write_idx_exp, key_caches[layer_index], val_caches[layer_index], attn_mask
                    )
                hidden = self.model.norm(hidden)
                logits_out.copy_(self.lm_head(hidden[:, -1, :]))

            write_idx_exp.fill_(max_len - 1)

            def replay_step(next_token_id: int, rope_offset: int, token_offset: int) -> None:
                token_buf.fill_(next_token_id)
                position_ids_buf.fill_(rope_offset)
                attn_mask[0, 0, 0, token_offset] = 0.0
                write_idx_exp.fill_(token_offset)
                graph.replay()
        else:
            start_tensor = torch.zeros((), dtype=torch.int32, device=device)
            start_index_tensor = torch.zeros((1,), dtype=torch.long, device=device)
            total_len_tensor = torch.zeros((), dtype=torch.int32, device=device)
            layer_caches = [
                self.state.kv_cache.get_kv_for_attention(layer_index) for layer_index in range(self.num_layers)
            ]
            tq_bufs = [
                TurboQuantGraphBuffers.create(layer_caches[layer_index], num_heads, max_len)
                for layer_index in range(self.num_layers)
            ]
            total_len_tensor.fill_(initial_kv_len)
            start_tensor.fill_(max_len - 1)
            start_index_tensor.fill_(max_len - 1)

            def _step():
                hidden = self.model.embed_tokens(token_buf)
                pos_emb = self._rotary_position_embeddings(position_ids_buf)
                for layer_index in range(self.num_layers):
                    hidden = self.model.layers[layer_index]._graph_forward_turboquant(
                        hidden,
                        pos_emb,
                        layer_caches[layer_index],
                        start_tensor,
                        start_index_tensor,
                        attn_mask,
                        tq_bufs[layer_index],
                        total_len_tensor,
                        max_len,
                    )
                hidden = self.model.norm(hidden)
                logits_out.copy_(self.lm_head(hidden[:, -1, :]))

            def replay_step(next_token_id: int, rope_offset: int, token_offset: int) -> None:
                token_buf.fill_(next_token_id)
                position_ids_buf.fill_(rope_offset)
                attn_mask[0, 0, 0, token_offset] = 0.0
                start_tensor.fill_(token_offset)
                start_index_tensor.fill_(token_offset)
                total_len_tensor.fill_(token_offset + 1)
                graph.replay()

        logger.info("[decode-graph] warming up (3 iters)...")
        for _ in range(3):
            _step()
        torch.cuda.synchronize()
        logger.info("[decode-graph] capturing graph...")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _step()
        logger.info("[decode-graph] ready")
        return DecodeGraphRunner(logits_out=logits_out, replay_step=replay_step)

    def load_misc(self):
        with tracer.timer("load_misc"):
            assert not self.misc_loaded
            tensors = self.model_loader.load_misc()
            if self.config.tie_word_embeddings:
                tensors = {name: tensor for name, tensor in tensors.items() if not name.startswith("lm_head.")}
            self._load_tensors(tensors)
            if self.config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
            self.misc_loaded = True

    def load_visual(self):
        with tracer.timer("load_visual"):
            assert not self.visual_loaded
            if self.blank_visual_marlin:
                self._load_blank_visual_for_marlin_benchmark()
            else:
                self._load_tensors(self.model_loader.load_visual())
            self.visual_loaded = True

    def load_layer(self, layer_idx: int):
        with tracer.timer("load_layer", layer_idx=layer_idx):
            assert layer_idx not in self.loaded_layers
            tensors = self.model_loader.load_layer(layer_idx)
            self._load_tensors(tensors)
            self._post_init_quantized_modules(self.model.layers[layer_idx])
            self.loaded_layers.add(layer_idx)

    def ensure_misc(self) -> None:
        if not self.misc_loaded:
            self.load_misc()

    def ensure_visual(self) -> None:
        if not self.visual_loaded:
            self.load_visual()

    def ensure_layer(self, layer_idx: int) -> None:
        if layer_idx not in self.loaded_layers:
            self.load_layer(layer_idx)

    def ensure_all_layers(self) -> None:
        for layer_idx in range(self.num_layers):
            self.ensure_layer(layer_idx)

    def _needs_visual_weights(self) -> bool:
        if self.state.phase != Phase.PREPARE_PREFILL:
            return False
        return (
            self.state.vision_checkpoint_count > 0
            or self.state.pixel_values is not None
            or self.state.pixel_values_videos is not None
        )

    def load_weights(self):
        with tracer.timer("load_weights", print_on_exit=True):
            self.ensure_misc()
            if self._needs_visual_weights():
                self.ensure_visual()
            self.ensure_all_layers()

    def _load_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        quantized_states: dict[str, dict[str, torch.Tensor]] = {}
        for name, tensor in tensors.items():
            module_path, attr_name = name.rsplit(".", 1)
            module = self.get_submodule(module_path)
            state_dict_keys = getattr(module, "state_dict_keys", None)
            if callable(state_dict_keys) and attr_name in state_dict_keys():
                quantized_states.setdefault(module_path, {})[attr_name] = tensor
                continue
            if attr_name in module._parameters:
                module._parameters[attr_name] = nn.Parameter(tensor, requires_grad=False)
            elif attr_name in module._buffers:
                module._buffers[attr_name] = tensor
            else:
                raise KeyError(f"tensor {module}.{attr_name} not found in model")
        for module_path, state in quantized_states.items():
            self.get_submodule(module_path).load_state_dict(dict(state), strict=True)

    def _post_init_quantized_modules(self, module: nn.Module) -> None:
        if self.text_quant_config is None:
            return
        from gptqmodel.nn_modules.qlinear import BaseQuantLinear

        for submodule in module.modules():
            if isinstance(submodule, BaseQuantLinear):
                submodule.post_init()

    def _load_blank_visual_for_marlin_benchmark(self) -> None:
        from gptqmodel.nn_modules.qlinear import BaseQuantLinear

        self.visual.to_empty(device=self._device)
        self.visual.to(dtype=torch.bfloat16)
        with torch.no_grad():
            for module in self.visual.modules():
                if isinstance(module, BaseQuantLinear):
                    module.qweight.zero_()
                    module.scales.fill_(1)
                    if getattr(module, "qzeros", None) is not None:
                        module.qzeros.zero_()
                    if getattr(module, "g_idx", None) is not None:
                        module.g_idx.zero_()
                    if getattr(module, "bias", None) is not None:
                        module.bias.zero_()
                    continue
                for param in module.parameters(recurse=False):
                    param.zero_()
                for buffer in module.buffers(recurse=False):
                    buffer.zero_()
        self._post_init_quantized_modules(self.visual)

    def minicpm_inputs_embeds(
        self,
        *,
        input_ids: torch.Tensor,
        pixel_values: list[list[torch.Tensor]] | None,
        image_bound: list[torch.Tensor] | None,
        tgt_sizes: list[torch.Tensor] | None,
        temporal_ids: list | None,
    ) -> torch.Tensor:
        assert is_minicpmv_config(self.config)
        input_ids = input_ids.to(device=self._device, dtype=torch.long)
        inputs_embeds = self.model.embed_tokens(input_ids)
        if not pixel_values:
            return inputs_embeds

        assert self.visual_loaded
        assert image_bound is not None and tgt_sizes is not None
        vision_hidden_states = self._minicpm_vision_hidden_states(pixel_values, tgt_sizes, temporal_ids)
        updated = torch.empty_like(inputs_embeds)
        for batch_idx, vision_hidden in enumerate(vision_hidden_states):
            if len(vision_hidden) == 0:
                updated[batch_idx] = inputs_embeds[batch_idx]
                continue
            bounds = image_bound[batch_idx].to(self._device)
            image_indices = torch.cat(
                [torch.arange(int(start), int(end), device=self._device) for start, end in bounds]
            )
            indices = image_indices.view(-1, 1).expand(-1, inputs_embeds.shape[-1])
            item = inputs_embeds[batch_idx].clone()
            item.scatter_(0, indices, vision_hidden.reshape(-1, inputs_embeds.shape[-1]).to(item.device, item.dtype))
            updated[batch_idx] = item
        return updated

    def _minicpm_vision_hidden_states(
        self,
        pixel_values_list: list[list[torch.Tensor]],
        tgt_sizes_list: list[torch.Tensor],
        temporal_ids: list | None,
    ) -> list[torch.Tensor | list]:
        dtype = self.model.embed_tokens.weight.dtype
        all_pixel_values = []
        image_counts = []
        for pixel_values in pixel_values_list:
            image_counts.append(len(pixel_values))
            all_pixel_values.extend(
                tensor.flatten(end_dim=1).permute(1, 0).to(device=self._device, dtype=dtype) for tensor in pixel_values
            )
        if not all_pixel_values:
            return [[] for _ in pixel_values_list]

        tgt_sizes = torch.vstack([item for item in tgt_sizes_list if isinstance(item, torch.Tensor)])
        tgt_sizes = tgt_sizes.to(device=self._device, dtype=torch.int32)
        max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])
        padded = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
        batch, length, _ = padded.shape
        padded = padded.permute(0, 2, 1).reshape(batch, 3, -1, length)
        patch_mask = torch.zeros((batch, 1, max_patches), dtype=torch.bool, device=self._device)
        for idx in range(batch):
            patch_mask[idx, 0, : tgt_sizes[idx][0] * tgt_sizes[idx][1]] = True

        vision_batch_size = int(self.config.vision_batch_size)
        chunks = []
        for start in range(0, batch, vision_batch_size):
            end = start + vision_batch_size
            chunks.append(
                self.minicpm_vpm(
                    padded[start:end],
                    patch_attention_mask=patch_mask[start:end],
                    tgt_sizes=tgt_sizes[start:end],
                ).last_hidden_state
            )
        vision_embedding = torch.cat(chunks, dim=0)
        vision_embedding = self.minicpm_resampler(
            vision_embedding,
            tgt_sizes,
            self._minicpm_temporal_ids(temporal_ids),
        )

        result = []
        start = 0
        for count in image_counts:
            if count:
                result.append(vision_embedding[start : start + count])
                start += count
            else:
                result.append([])
        return result

    @staticmethod
    def _minicpm_temporal_ids(temporal_ids: list | None) -> list | None:
        if temporal_ids is None:
            return None

        def convert(value):
            if isinstance(value, torch.Tensor):
                return int(value.item())
            if isinstance(value, list):
                return [convert(item) for item in value]
            return int(value)

        result = []
        for item in temporal_ids:
            result.extend(convert(item))
        return result

    def release_visual(self) -> None:
        self.state.pixel_values = None
        self.state.pixel_values_videos = None
        del self.visual

    def _materialize_derived_buffers(self) -> None:
        """Recompute non-persistent buffers (rotary freqs) that are meta tensors after meta-device init."""
        self.rotary_inv_freq = build_rotary_inv_freq(self.config.text_config).to(self._device)
        self.visual.rotary_inv_freq = build_vision_rotary_inv_freq(self.config.vision_config).to(self._device)
        if is_minicpmv_config(self.config):
            self.minicpm_resampler.pos_embed = self.minicpm_resampler.pos_embed.to(self._device)
            self.minicpm_resampler.temporal_pos_embed = self.minicpm_resampler.temporal_pos_embed.to(self._device)

    @torch.no_grad()
    def _rotary_position_embeddings(self, position_ids: torch.Tensor) -> RotaryEmbedding:
        """Compute rotary position embeddings.

        Args:
            position_ids: (1, length) for standard 1D RoPE, or (3, 1, length) for MRoPE.
        Returns:
            Rotary embeddings with `cos` and `sin` of shape `(1, length, head_dim)`.
        """
        inv_freq = self.rotary_inv_freq.to(device=self._device)
        with maybe_autocast(device_type=self._device.type, enabled=False):
            if position_ids.ndim == 2:
                # Standard 1D RoPE
                inv_freq_expanded = inv_freq[None, :, None]  # (1, dim//2, 1)
                pos_expanded = position_ids[:, None, :].float()  # (1, 1, length)
                freqs = (inv_freq_expanded @ pos_expanded).transpose(1, 2)  # (1, length, dim//2)
            else:
                # MRoPE: position_ids is (3, 1, length)
                mrope_section = self.config.text_config.rope_parameters["mrope_section"]  # type: ignore[index]
                bs = position_ids.shape[1]
                inv_freq_expanded = inv_freq[None, None, :, None].expand(3, bs, -1, 1)  # (3, 1, dim//2, 1)
                pos_expanded = position_ids[:, :, None, :].float()  # (3, 1, 1, length)
                freqs = (inv_freq_expanded @ pos_expanded).transpose(2, 3)  # (3, 1, length, dim//2)
                freqs = apply_interleaved_mrope(freqs, mrope_section)  # (1, length, dim//2)
        return RotaryEmbedding.from_freqs(freqs)

    def get_visual_features(
        self, input_idx: int, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> BaseModelOutputWithDeepstackFeatures:
        """Run the vision tower and split merged outputs back into per-item chunks.

        Args:
            pixel_values: Visual patch tensor of shape `(total_visual_tokens, patch_dim)`, where
                `total_visual_tokens = sum(t * h * w)` across `grid_thw`.
            grid_thw: Long tensor of shape `(num_visual_items, 3)`. Each row is `[t, h, w]` for one
                image or video sample in vision-grid coordinates.

        Returns:
            `BaseModelOutputWithDeepstackFeatures` where:
            - `pooler_output` is a tuple of length `num_visual_items`; entry `i` has shape
              `(t_i * h_i * w_i // spatial_merge_size**2, hidden_size)`.
            - `deepstack_features` is a list over selected vision layers; each tensor has shape
              `(sum(t * h * w) // spatial_merge_size**2, hidden_size)`.
            - `last_hidden_state` keeps the raw vision-tower hidden states.
        """
        pixel_values = pixel_values.to(device=self._device, dtype=self.visual.dtype)
        grid_thw = grid_thw.to(self._device)
        cu_seqlens, pos_emb = self.visual.prepare_layout(grid_thw)
        assert input_idx <= self.state.vision_checkpoint_count
        if input_idx == self.state.vision_checkpoint_count:
            hidden_states = self.visual.prepare_stem(pixel_values, grid_thw)
            self.state.init_vision_checkpoint(input_idx, hidden_states, len(self.visual.blocks))
            start_layer = 0
        else:
            start_layer = self.state.vision_layer_progress(input_idx)
            hidden_states = self.state.load_vision_stage(input_idx, start_layer, self._device)

        for layer_num in range(start_layer, len(self.visual.blocks)):
            hidden_states = self.visual.run_layer(layer_num, hidden_states, cu_seqlens, pos_emb)
            self.state.save_vision_layer(input_idx, layer_num, hidden_states)

        deepstack_features = []
        for idx, layer_num in enumerate(self.visual.deepstack_visual_indexes):
            layer_hidden = self.state.load_vision_stage(input_idx, layer_num + 1, self._device)
            deepstack_features.append(self.visual.deepstack_merger_list[idx](layer_hidden))

        vision_output = BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=self.visual.merger(hidden_states),
            deepstack_features=deepstack_features,
        )
        split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_output.pooler_output = torch.split(vision_output.pooler_output, split_sizes)
        return vision_output

    def _apply_visual_inputs_incremental(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        visual_inputs: list[VisualInput],
        visual_pool: torch.cuda.MemPool | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        assert input_ids.shape[0] == 1

        visual_pos_masks = torch.zeros_like(input_ids, dtype=torch.bool)
        for visual_input in visual_inputs:
            visual_pos_masks |= input_ids == visual_input.token_id

        visual_items = []
        spatial_merge = self.config.vision_config.spatial_merge_size
        for visual_input in visual_inputs:
            modality_positions = (input_ids[0] == visual_input.token_id).nonzero(as_tuple=False).squeeze(1)
            token_offset = 0
            patch_offset = 0
            llm_grid_offset = 0
            for grid_thw in visual_input.vision_grid_thw.split(1):
                llm_grid_count = int(grid_thw[0, 0].item())
                llm_grid_thw = visual_input.llm_grid_thw[llm_grid_offset : llm_grid_offset + llm_grid_count]
                token_count = int((llm_grid_thw.prod(-1) // spatial_merge**2).sum().item())
                patch_count = int(grid_thw.prod().item())
                positions = modality_positions[token_offset : token_offset + token_count]
                pixel_values = visual_input.pixel_values.narrow(0, patch_offset, patch_count)
                visual_items.append((int(positions[0]), positions, pixel_values, grid_thw))
                token_offset += token_count
                patch_offset += patch_count
                llm_grid_offset += llm_grid_count
            assert token_offset == int(modality_positions.numel())

        visual_items.sort(key=lambda item: item[0])
        deepstack_visual_embeds: list[torch.Tensor] | None = None
        visual_cursor = 0
        for input_idx, (_, item_positions, pixel_values, grid_thw) in enumerate(visual_items):
            logger.info("[vit] visual %d/%d", input_idx + 1, len(visual_items))
            pool_context = (
                torch.cuda.use_mem_pool(visual_pool, torch.cuda.current_device())
                if visual_pool is not None and self._device.type == "cuda"
                else nullcontext()
            )
            with pool_context:
                visual_outputs = self.get_visual_features(input_idx, pixel_values, grid_thw)

            assert len(visual_outputs.pooler_output) == 1
            visual_embeds = visual_outputs.pooler_output[0]
            item_len = int(visual_embeds.shape[0])
            assert item_len == int(item_positions.numel())

            inputs_embeds[0].index_copy_(
                0,
                item_positions,
                visual_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
            )

            if deepstack_visual_embeds is None:
                visual_count = int(visual_pos_masks.sum().item())
                deepstack_visual_embeds = [
                    inputs_embeds.new_empty((visual_count, layer_hidden.shape[-1]))
                    for layer_hidden in visual_outputs.deepstack_features
                ]

            for layer_idx, layer_hidden in enumerate(visual_outputs.deepstack_features):
                deepstack_visual_embeds[layer_idx][visual_cursor : visual_cursor + item_len].copy_(
                    layer_hidden.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                )
            visual_cursor += item_len

        return visual_pos_masks, deepstack_visual_embeds or []

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_mask: torch.Tensor,
        visual_embeds: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        chunk_visual_mask = visual_pos_mask[start:end].to(hidden_states.device)
        visual_count = int(chunk_visual_mask.sum().item())
        if visual_count == 0:
            return hidden_states

        visual_offset = int(visual_pos_mask[:start].sum().item())
        visual_embeds = visual_embeds[visual_offset : visual_offset + visual_count]
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[0, chunk_visual_mask] = hidden_states[0, chunk_visual_mask] + visual_embeds
        return hidden_states

    def _prefill_forward_chunk_layer(
        self,
        idx: ChunkLayerIndex,
        hidden_states: torch.Tensor,
        pos_emb: RotaryEmbedding,
    ) -> torch.Tensor:
        with tracer.timer("prefill_chunk_layer", layer_idx=idx.layer, chunk_id=idx.chunk_id):
            decoder_layer = self.model.layers[idx.layer]
            new_hidden = decoder_layer(
                hidden_states,
                r=Range(layer=idx.layer, start=idx.start, end=idx.end),
                pos_emb=pos_emb,
                generation_state=self.state,
            )
        return new_hidden

    def _decode_forward_token(self, token_id: int, token_offset: int, rope_offset: int) -> torch.Tensor:
        with tracer.timer("decode_forward_token", offset=token_offset):
            input_ids = torch.tensor([[token_id]], device=self._device, dtype=torch.long)
            hidden_states = self.model.embed_tokens(input_ids)
            position_ids = torch.full((3, 1, 1), rope_offset, device=self._device, dtype=torch.long)
            pos_emb = self._rotary_position_embeddings(position_ids)
            for layer_idx in range(self.num_layers):
                decoder_layer = self.model.layers[layer_idx]
                hidden_states = decoder_layer(
                    hidden_states=hidden_states,
                    r=Range(layer=layer_idx, start=token_offset, end=token_offset + 1),
                    pos_emb=pos_emb,
                    generation_state=self.state,
                )
            hidden_states = self.model.norm(hidden_states)
        return hidden_states

    def _run_decode_graph(self, on_token: Callable[[int], None] | None = None) -> list[int]:
        """Decode using a captured CUDA graph to amortize kernel-launch overhead."""
        max_decode_tokens = self.runtime_config.max_decode_tokens

        already_done = len(self.state.generated_tokens) - 1
        token_id = self.state.generated_tokens[-1]
        runner = self._capture_decode_graph_runner()
        if self.state.phase == Phase.DECODING_DENSE_OVERLAP:
            self._ensure_pending_turboquant_future()

        timer = tracer.timer("decode", mode="cuda_graph").start()
        last_ngen = 0
        last_ms = timer.elapsed_ms()

        while len(self.state.generated_tokens) < max_decode_tokens:
            if (
                self.runtime_config.stop_after_tokens is not None
                and len(self.state.generated_tokens) >= self.runtime_config.stop_after_tokens
            ):
                break
            if on_token is not None:
                on_token(token_id)
            if token_id == self.config.text_config.eos_token_id:
                break

            token_offset = self.state.prompt_len + already_done
            rope_offset = self.state.decode_rope_offset + already_done
            runner.replay_step(token_id, rope_offset, token_offset)
            token_id = self.sampler.sample_next_token(runner.logits_out)
            self.state.advance_decode(token_id)
            self.state.commit()
            already_done += 1

            if self.state.phase == Phase.DECODING_DENSE_OVERLAP and self._pending_turboquant_future is not None:
                if self._pending_turboquant_future.done():
                    self._switch_to_turboquant()
                    torch.cuda.synchronize()
                    started_at = perf_counter()
                    runner = self._capture_decode_graph_runner()
                    torch.cuda.synchronize()
                    elapsed_ms = (perf_counter() - started_at) * 1000.0
                    self._add_phase_timing_ms("kv_cache_compress_graph_recapture", elapsed_ms)
                    self._add_phase_timing_ms("kv_cache_compress", elapsed_ms)

            now_ms = timer.elapsed_ms()
            if now_ms - last_ms >= 1000.0:
                elapsed = (now_ms - last_ms) / 1000.0
                n_gen = len(self.state.generated_tokens)
                tps = (n_gen - last_ngen) / elapsed if elapsed > 0.0 else 0.0
                logger.info("[decode] %d/%d tok, %.2fs, %.1f tok/s", n_gen, max_decode_tokens, elapsed, tps)
                last_ngen, last_ms = n_gen, now_ms

        if self.state.phase == Phase.DECODING_DENSE_OVERLAP and self._pending_turboquant_future is not None:
            self._switch_to_turboquant()
        timer.end()
        return self.state.generated_tokens

    def _run_decode(self, on_token: Callable[[int], None] | None = None) -> list[int]:
        if self.runtime_config.cuda_graph:
            return self._run_decode_graph(on_token=on_token)
        assert not self.runtime_config.turboquant_overlap_decode

        already_done = len(self.state.generated_tokens) - 1
        max_decode_tokens = self.runtime_config.max_decode_tokens
        token_id = self.state.generated_tokens[-1]
        sync_cuda_timers = self.runtime_config.sync_cuda_timers

        timer = tracer.timer("decode", mode="sync", sync_cuda=sync_cuda_timers).start()
        last_ngen = 0
        last_ms = timer.elapsed_ms()
        while len(self.state.generated_tokens) < max_decode_tokens:
            if (
                self.runtime_config.stop_after_tokens is not None
                and len(self.state.generated_tokens) >= self.runtime_config.stop_after_tokens
            ):
                break
            if on_token is not None:
                on_token(token_id)
            if token_id == self.config.text_config.eos_token_id:
                break
            token_offset = self.state.prompt_len + already_done
            rope_offset = self.state.decode_rope_offset + already_done
            with tracer.timer("decode_phase_forward", sync_cuda=sync_cuda_timers):
                hidden_states = self._decode_forward_token(token_id, token_offset, rope_offset)
            with tracer.timer("decode_phase_lm_head", sync_cuda=sync_cuda_timers):
                logits = self.lm_head(hidden_states[:, -1, :])
            with tracer.timer("decode_phase_sample", sync_cuda=sync_cuda_timers):
                token_id = self.sampler.sample_next_token(logits)
            with tracer.timer("decode_phase_advance", sync_cuda=sync_cuda_timers):
                self.state.advance_decode(token_id)
            with tracer.timer("decode_phase_commit", sync_cuda=sync_cuda_timers):
                self.state.commit()
            already_done += 1

            now_ms = timer.elapsed_ms()
            if now_ms - last_ms >= 1000.0:
                elapsed = (now_ms - last_ms) / 1000.0
                n_gen = len(self.state.generated_tokens)
                tps = (n_gen - last_ngen) / elapsed if elapsed > 0.0 else 0.0
                logger.info("[decode] %d/%d tok, %.2fs, %.1f tok/s", n_gen, max_decode_tokens, elapsed, tps)
                last_ngen, last_ms = n_gen, now_ms

        timer.end()
        return self.state.generated_tokens

    def _prefill_output(self, prompt_len: int) -> torch.Tensor:
        if self.runtime_config.prefill_order == "chunk_first":
            final_hidden = self.state.get_prefill_output_hidden()
        else:
            final_hidden = self.state.get_h(Range(layer=self.num_layers - 1, start=prompt_len - 1, end=prompt_len))
        assert final_hidden is not None
        return self.model.norm(final_hidden)

    @torch.inference_mode()
    def prefill_chunk_first_sync(
        self,
        inputs_embeds: torch.Tensor,
        pos_emb: RotaryEmbedding,
        visual_pos_masks: torch.Tensor,
        deepstack_visual_embeds: list[torch.Tensor],
    ) -> torch.Tensor:
        prompt_len = inputs_embeds.shape[1]
        timer = tracer.timer("prefill", mode="chunk_first").start()
        start_chunk = self.state.completed_chunks
        chunk_size = self.runtime_config.prefill_chunk_size
        num_chunks = (prompt_len + chunk_size - 1) // chunk_size
        for chunk_id in range(start_chunk, num_chunks):
            chunk_timer = tracer.timer("prefill-chunk", chunk_id=chunk_id).start()
            chunk_start = chunk_id * chunk_size
            chunk_end = min(chunk_start + chunk_size, prompt_len)
            chunk_pos_emb = pos_emb.slice(chunk_start, chunk_end)
            hidden_states = inputs_embeds[:, chunk_start:chunk_end, :]

            for layer_id in range(self.config.text_config.num_hidden_layers):
                idx = ChunkLayerIndex(chunk_id=chunk_id, layer=layer_id, start=chunk_start, end=chunk_end)
                with tracer.timer("prefill-chunk-layer", layer_id=layer_id, chunk_id=chunk_id):
                    hidden_states = self._prefill_forward_chunk_layer(idx, hidden_states, chunk_pos_emb)
                    if layer_id < len(deepstack_visual_embeds):
                        hidden_states = self._deepstack_process(
                            hidden_states,
                            visual_pos_masks[0],
                            deepstack_visual_embeds[layer_id],
                            start=chunk_start,
                            end=chunk_end,
                        )
                    self.state.advance_prefill(idx)
            if chunk_end == prompt_len:
                self.state.update_prefill_output_hidden(hidden_states[:, -1:, :])
            self.state.commit()
            chunk_timer.end()
            logger.info(
                "[prefill-chunk] chunk %d/%d, %d tok, %.2fms",
                chunk_id + 1,
                num_chunks,
                chunk_end - chunk_start,
                chunk_timer.elapsed_ms(),
            )
        timer.end()
        return self._prefill_output(prompt_len)

    @torch.inference_mode()
    def prefill_scheduled_sync(
        self,
        inputs_embeds: torch.Tensor,
        pos_emb: RotaryEmbedding,
        visual_pos_masks: torch.Tensor,
        deepstack_visual_embeds: list[torch.Tensor],
        strategy: PrefillScheduleStrategy | None = None,
    ) -> torch.Tensor:
        strategy = strategy or build_prefill_strategy(self.runtime_config.prefill_schedule_strategy)
        scheduler = PrefillScheduler(
            self,
            PrefillInputs(
                inputs_embeds, pos_emb, visual_pos_masks, deepstack_visual_embeds, self.state.decode_rope_offset
            ),
            strategy,
        )
        return scheduler.run()

    @torch.inference_mode()
    @tracer.timed("prefill", end_name="prefill", mode="layer_first")
    def prefill_layer_first_sync(
        self,
        inputs_embeds: torch.Tensor,
        pos_emb: RotaryEmbedding,
        visual_pos_masks: torch.Tensor,
        deepstack_visual_embeds: list[torch.Tensor],
    ) -> torch.Tensor:
        prompt_len = inputs_embeds.shape[1]

        start_layer = self.state.completed_layers
        if start_layer == 0:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.state.get_h(Range(layer=start_layer - 1, start=0, end=prompt_len))
            assert hidden_states is not None

        for layer_id in range(start_layer, self.config.text_config.num_hidden_layers):
            idx = ChunkLayerIndex(chunk_id=self.state.total_chunks - 1, layer=layer_id, start=0, end=prompt_len)
            hidden_states = self._prefill_forward_chunk_layer(idx, hidden_states, pos_emb)
            if layer_id < len(deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks[0],
                    deepstack_visual_embeds[layer_id],
                    start=0,
                    end=prompt_len,
                )
            self.state.update_h(Range(layer=layer_id, start=0, end=prompt_len), hidden_states)
            self.state.advance_prefill(idx)
            self.state.commit()
            num_layers = self.config.text_config.num_hidden_layers
            logger.info("[prefill-layer] layer %d/%d, %d tok", layer_id + 1, num_layers, prompt_len)
        return self._prefill_output(prompt_len)

    def _run_prefill(self, prefill_inputs: PrefillInputs) -> torch.Tensor:
        prompt_len = prefill_inputs.inputs_embeds.shape[1]
        if self.state.completed_layers == self.num_layers:
            return self._prefill_output(prompt_len)
        args = (
            prefill_inputs.inputs_embeds,
            prefill_inputs.pos_emb,
            prefill_inputs.visual_pos_masks,
            prefill_inputs.deepstack_visual_embeds,
        )
        if self.runtime_config.prefill_order == "chunk_first":
            return self.prefill_chunk_first_sync(*args)
        if self.runtime_config.prefill_order == "layer_first":
            return self.prefill_layer_first_sync(*args)
        if self.runtime_config.prefill_order == "scheduled":
            return self.prefill_scheduled_sync(*args)
        raise ValueError(f"Invalid prefill order: {self.runtime_config.prefill_order}")

    def _build_prefill_inputs(self, visual_pool: torch.cuda.MemPool | None = None) -> PrefillInputs:
        input_ids = self.state.input_ids
        inputs_embeds = self.model.embed_tokens(input_ids)
        visual_inputs = build_visual_inputs(self.state)
        if not visual_inputs:
            position_ids = torch.arange(0, input_ids.shape[1], device=self._device, dtype=torch.long).unsqueeze(0)
            pos_emb = self._rotary_position_embeddings(position_ids)
            return PrefillInputs(
                inputs_embeds,
                pos_emb,
                torch.zeros_like(input_ids, dtype=torch.bool),
                [],
                input_ids.shape[1],
            )

        with self._phase_timer("vit"):
            with tracer.timer("apply_visual_inputs_incremental", print_on_exit=True):
                visual_pos_masks, deepstack_visual_embeds = self._apply_visual_inputs_incremental(
                    input_ids, inputs_embeds, visual_inputs, visual_pool
                )
        self.release_visual()
        gc.collect()

        vision_token_count = int(visual_pos_masks.sum().item())
        logger.info(
            "[prepare-input] vision_tokens=%d prompt_len=%d",
            vision_token_count,
            int(input_ids.shape[1]),
        )

        position_ids, rope_deltas = compute_rope_index(
            input_ids=input_ids,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            vision_start_token_id=self.config.vision_start_token_id,
            visual_inputs=visual_inputs,
        )
        decode_rope_offset = input_ids.shape[1] + int(rope_deltas[0, 0].item())

        pos_emb = self._rotary_position_embeddings(position_ids)
        return PrefillInputs(
            inputs_embeds,
            pos_emb,
            visual_pos_masks,
            deepstack_visual_embeds,
            decode_rope_offset,
        )

    @torch.inference_mode()
    def generate(self, on_token: Callable[[int], Any] | None = None) -> list[int]:
        self.last_phase_timings_ms = {}
        self.last_phase_memory_mb = {}
        prefill_inputs = None
        if self.state.phase == Phase.PREPARE_PREFILL:
            self.ensure_misc()
            if self._needs_visual_weights():
                if self._device.type == "cuda":
                    visual_pool = torch.cuda.MemPool()
                    with torch.cuda.use_mem_pool(visual_pool, torch.cuda.current_device()):
                        self.ensure_visual()
                    prefill_inputs = self._build_prefill_inputs(visual_pool=visual_pool)
                    del visual_pool
                    gc.collect()
                else:
                    self.ensure_visual()
                    prefill_inputs = self._build_prefill_inputs()
            else:
                prefill_inputs = self._build_prefill_inputs()
            self._log_prefill_memory_model(prefill_inputs)
            self.state.commit_prefill_inputs(prefill_inputs)

        if self.state.phase == Phase.PREFILLING:
            self.ensure_misc()
            if self.runtime_config.prefill_order != "scheduled":
                self.ensure_all_layers()
            if prefill_inputs is None:
                prefill_inputs = self.state.load_prefill_inputs()
            with self._phase_timer("prefill"):
                prefill_hidden = self._run_prefill(prefill_inputs)
                first_token_id = self.sampler.sample_next_token(self.lm_head(prefill_hidden[:, -1, :]))
            self.state.begin_prepare_decoding(first_token_id)
            prefill_inputs = None

        if self.state.phase == Phase.PREPARE_DECODING:
            with self._phase_timer("kv_cache_compress"):
                self.state.prepare_decoding()

        if self.state.phase in {Phase.DECODING_DENSE_OVERLAP, Phase.DECODING}:
            self.ensure_misc()
            self.ensure_all_layers()
            with self._phase_timer("decode"):
                tokens = self._run_decode(on_token=on_token)
            return tokens

        raise ValueError(f"Unexpected generation phase: {self.state.phase}")
