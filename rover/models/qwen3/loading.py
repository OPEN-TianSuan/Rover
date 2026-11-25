"""Model directory configuration."""

from rover.models.qwen3.config import Qwen3Config


def load_model_config(model_dir):
    config = Qwen3Config.from_pretrained(model_dir)
    if config.model_type != "qwen3_vl":
        raise ValueError("Rover currently supports Qwen3-VL dense weights")
    return config


def is_minicpmv_config(config):
    return getattr(config, "source_model_type", None) == "minicpmv"


def minicpmv_vision_classes(model_dir):
    raise NotImplementedError("MiniCPM vision is not supported")
