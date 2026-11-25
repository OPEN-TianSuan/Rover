"""Local JSON model settings."""

import json
from pathlib import Path


class Config:
    def __init__(self, **kwargs):
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, data):
        return cls(**dict(data))

    @classmethod
    def from_pretrained(cls, path, *, local_files_only=True):
        path = Path(path).expanduser()
        if path.is_dir():
            path /= "config.json"
        return cls.from_dict(json.loads(path.read_text()))

    def to_dict(self):
        def encode(value):
            if isinstance(value, Config):
                return value.to_dict()
            if isinstance(value, dict):
                return {k: encode(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(v) for v in value]
            return value

        return {"model_type": self.model_type, **{k: encode(v) for k, v in vars(self).items()}}
