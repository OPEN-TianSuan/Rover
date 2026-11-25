"""Local chat formatting, tokenization and image patch preparation."""

import json
import math
from pathlib import Path

import torch


class Processor:
    def __init__(self, directory):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        self.directory = Path(directory).expanduser()
        self.tokenizer = Tokenizer.from_file(str(self.directory / "tokenizer.json"))
        settings = json.loads((self.directory / "tokenizer_config.json").read_text())
        template_path = self.directory / "chat_template.jinja"
        if template_path.exists():
            template = template_path.read_text()
        elif (self.directory / "chat_template.json").exists():
            template = json.loads((self.directory / "chat_template.json").read_text())["chat_template"]
        else:
            template = settings["chat_template"]
        self.template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(template)
        self.image_settings = json.loads((self.directory / "preprocessor_config.json").read_text())

    def decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def image(self, path):
        import numpy as np
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode, resize

        spec = self.image_settings
        patch = spec["patch_size"]
        merge = spec["merge_size"]
        factor = patch * merge
        with Image.open(path) as source:
            pixels = torch.from_numpy(np.array(source.convert("RGB"))).permute(2, 0, 1)
        _, height, width = pixels.shape
        if max(height, width) / min(height, width) > 200:
            raise ValueError("Image aspect ratio exceeds 200")
        new_h = max(factor, round(height / factor) * factor)
        new_w = max(factor, round(width / factor) * factor)
        lower, upper = spec["size"]["shortest_edge"], spec["size"]["longest_edge"]
        if new_h * new_w < lower:
            ratio = math.sqrt(lower / (height * width))
            new_h, new_w = [math.ceil(side * ratio / factor) * factor for side in (height, width)]
        elif new_h * new_w > upper:
            ratio = math.sqrt(upper / (height * width))
            new_h, new_w = [max(factor, math.floor(side * ratio / factor) * factor) for side in (height, width)]
        pixels = resize(pixels, [new_h, new_w], interpolation=InterpolationMode.BICUBIC, antialias=True).float() / 255
        mean = torch.tensor(spec["image_mean"])[:, None, None]
        std = torch.tensor(spec["image_std"])[:, None, None]
        pixels = (pixels - mean) / std
        gh, gw = new_h // patch, new_w // patch
        temporal = spec["temporal_patch_size"]
        frames = pixels.unsqueeze(0).expand(temporal, -1, -1, -1)
        # Spatial groups are contiguous so each merged group becomes one language token.
        blocks = frames.reshape(temporal, 3, gh // merge, merge, patch, gw // merge, merge, patch)
        blocks = blocks.permute(2, 5, 3, 6, 1, 0, 4, 7).contiguous()
        return blocks.reshape(gh * gw, 3 * temporal * patch * patch), [1, gh, gw]

    def encode(self, prompt, images=()):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.template.render(
            messages=[{"role": "user", "content": content}], add_generation_prompt=True, tools=None
        )
        tensors, grids = [], []
        marker = "<|image_pad|>"
        parts = text.split(marker)
        if len(parts) != len(images) + 1:
            raise ValueError("Chat template image placeholders do not match image inputs")
        text = parts[0]
        for index, path in enumerate(images):
            pixels, grid = self.image(path)
            tensors.append(pixels)
            grids.append(grid)
            count = grid[1] * grid[2] // self.image_settings["merge_size"] ** 2
            text += marker * count + parts[index + 1]
        tokens = self.tokenizer.encode(text, add_special_tokens=False).ids
        result = {"input_ids": torch.tensor([tokens], dtype=torch.long)}
        if tensors:
            result.update(pixel_values=torch.cat(tensors), image_grid_thw=torch.tensor(grids))
        return result
