from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SamplingParams:
    do_sample: bool
    temperature: float
    top_k: int
    top_p: float
    seed: int


DEFAULT_SAMPLING_PARAMS = SamplingParams(
    do_sample=False,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    seed=0,
)


@dataclass(frozen=True)
class RuntimeConfig:
    model_dir: str
    device: str
    state_dir: str | None
    stop_after_tokens: int | None
    max_decode_tokens: int
    max_cache_len: int
    max_batch_size: int
    prefill_chunk_size: int
    prefill_order: str
    sampling: SamplingParams
    prefill_schedule_strategy: str = field(default="chunk_first")
    prefill_schedule_layer_group_size: int = field(default=1)
    kv_cache_format: str = field(default="dense")
    kivi_bits: int = field(default=2)
    kivi_group_size: int = field(default=32)
    kivi_residual_length: int = field(default=128)
    turboquant_bits: int = field(default=3)
    turboquant_outlier_bits: int = field(default=4)
    turboquant_outlier_channels: int = field(default=32)
    turboquant_seed: int = field(default=0)
    turboquant_mode: str = field(default="mse")
    turboquant_overlap_decode: bool = field(default=False)
    profile: str = field(default="off")
    cuda_graph: bool = field(default=False)

    @property
    def trace_enabled(self) -> bool:
        return self.profile != "off"

    @property
    def sync_cuda_timers(self) -> bool:
        return self.profile in {"sync", "full"}

    @property
    def turboquant_profile(self) -> bool:
        return self.profile == "full"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: dict[str, object]) -> "RuntimeConfig":
        data = dict(obj)
        assert "profile" in data
        data["sampling"] = SamplingParams(**data["sampling"])
        return cls(**data)

    def validate(self) -> None:
        if self.stop_after_tokens is not None and self.stop_after_tokens <= 0:
            raise ValueError("stop_after_tokens must be positive")
        if self.cuda_graph and not self.device.startswith("cuda"):
            raise ValueError("cuda_graph requires a CUDA device")
        assert self.prefill_order in {"chunk_first", "layer_first", "scheduled"}
        assert self.prefill_schedule_strategy in {"chunk_first", "layer_first", "greedy", "state_compression_dp"}
        assert self.prefill_schedule_layer_group_size > 0
        assert self.max_decode_tokens > 0
        assert self.prefill_chunk_size > 0
        assert self.device == "cpu" or self.device.startswith("cuda")
        assert self.sampling.temperature > 0.0
        assert self.sampling.top_k >= 0
        assert 0.0 < self.sampling.top_p <= 1.0
        assert self.max_cache_len > 0
        assert self.max_batch_size == 1
        assert self.kv_cache_format in {"dense", "kivi", "turboquant"}
        assert self.kivi_bits == 2
        assert self.kivi_group_size > 0
        assert self.kivi_residual_length > 0
        assert self.kivi_residual_length % self.kivi_group_size == 0
        assert self.turboquant_bits > 0
        assert self.turboquant_outlier_bits >= self.turboquant_bits
        assert self.turboquant_outlier_channels > 0
        assert self.turboquant_mode in {"mse", "prod"}
        if self.turboquant_mode == "prod":
            assert self.turboquant_bits >= 2
            assert self.turboquant_outlier_bits >= 2
        assert self.profile in {"off", "trace", "sync", "full"}
        if self.turboquant_overlap_decode:
            assert self.kv_cache_format == "turboquant"
            assert self.cuda_graph, "turboquant_overlap_decode requires cuda_graph"
        if self.cuda_graph:
            assert self.kv_cache_format in ("dense", "turboquant"), (
                "cuda_graph requires dense or turboquant kv_cache_format"
            )
            assert not self.trace_enabled, "cuda_graph is incompatible with profiling"

    def __post_init__(self) -> None:
        self.validate()
