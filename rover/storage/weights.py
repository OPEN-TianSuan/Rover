# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportMissingTypeArgument=false, reportMissingImports=false

import json
from pathlib import Path
from typing import Final

import torch
from safetensors import safe_open

from rover.diagnostics.tracing import tracer


def json_dict(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("expected json object")
    return value


class ModelLoader:
    INDEX_FILE: Final[str] = "model.safetensors.index.json"
    SINGLE_FILE: Final[str] = "model.safetensors"
    MISC_PREFIXES: Final[tuple[str, ...]] = (
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "llm.model.embed_tokens.",
        "llm.model.norm.",
        "llm.lm_head.",
        "lm_head.",
    )
    VISUAL_PREFIX: Final[str] = "model.visual."
    MINICPM_VISUAL_PREFIXES: Final[tuple[str, ...]] = ("vpm.", "resampler.")

    def __init__(self, model_dir: str, device: torch.device):
        self.model_dir = Path(model_dir)
        self.device = device
        self.tensor_to_file = self._build_tensor_to_file()
        self.layer_tensor_names: dict[int, list[str]] = {}
        self.misc_tensor_names: list[str] = []
        self.visual_tensor_names: list[str] = []
        self._group_tensor_names()

    def load_misc(self) -> dict[str, torch.Tensor]:
        with tracer.timer("model_loader_load_misc"):
            return self._load_names(self.misc_tensor_names)

    def load_visual(self) -> dict[str, torch.Tensor]:
        with tracer.timer("model_loader_load_visual"):
            return self._load_names(self.visual_tensor_names)

    def load_layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        names = self.layer_tensor_names.get(layer_idx)
        if not names:
            raise KeyError(f"no tensors found for layer {layer_idx}")
        with tracer.timer("model_loader_load_layer", layer_idx=int(layer_idx)):
            return self._load_names(names)

    def _build_tensor_to_file(self) -> dict[str, Path]:
        index_path = self.model_dir / self.INDEX_FILE
        if index_path.exists():
            index_json = json_dict(json.loads(index_path.read_text(encoding="utf-8")))
            weight_map = index_json.get("weight_map")
            assert isinstance(weight_map, dict), "invalid safetensors index: missing weight_map"
            return {name: self.model_dir / file_name for name, file_name in weight_map.items()}

        single_path = self.model_dir / self.SINGLE_FILE
        if not single_path.exists():
            raise FileNotFoundError(f"no safetensors weights found in {self.model_dir}")
        with safe_open(str(single_path), framework="pt") as handle:
            return dict.fromkeys(handle.keys(), single_path)

    def _group_tensor_names(self) -> None:
        for name in self.tensor_to_file:
            layer_idx = self._parse_layer_idx(name)
            if layer_idx is not None:
                self.layer_tensor_names.setdefault(layer_idx, []).append(name)
                continue
            if any(name.startswith(prefix) for prefix in self.MISC_PREFIXES):
                self.misc_tensor_names.append(name)
            elif name.startswith(self.VISUAL_PREFIX) or any(
                name.startswith(prefix) for prefix in self.MINICPM_VISUAL_PREFIXES
            ):
                self.visual_tensor_names.append(name)
        for names in self.layer_tensor_names.values():
            names.sort()
        self.misc_tensor_names.sort()
        self.visual_tensor_names.sort()

    def _load_names(self, names: list[str]) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        names_by_file: dict[Path, list[str]] = {}
        for name in names:
            file_path = self.tensor_to_file.get(name)
            assert file_path is not None, f"tensor not found in index: {name}"
            names_by_file.setdefault(file_path, []).append(name)

        for file_path, file_names in names_by_file.items():
            with safe_open(str(file_path), framework="pt") as handle:
                for name in file_names:
                    normalized = name.replace("model.language_model.", "model.", 1)
                    normalized = normalized.replace("model.visual.", "visual.", 1)
                    normalized = normalized.replace("llm.model.", "model.", 1)
                    normalized = normalized.replace("llm.lm_head.", "lm_head.", 1)
                    normalized = normalized.replace("vpm.", "minicpm_vpm.", 1)
                    normalized = normalized.replace("resampler.", "minicpm_resampler.", 1)
                    tensors[normalized] = handle.get_tensor(name).to(device=self.device, non_blocking=False)
        return tensors

    @staticmethod
    def _parse_layer_idx(name: str) -> int | None:
        if name.startswith("model.language_model.layers."):
            parts = name.split(".")
            idx_part = parts[3] if len(parts) > 3 else None
        elif name.startswith("llm.model.layers."):
            parts = name.split(".")
            idx_part = parts[3] if len(parts) > 3 else None
        else:
            return None
        try:
            return int(idx_part) if idx_part is not None else None
        except ValueError:
            return None
