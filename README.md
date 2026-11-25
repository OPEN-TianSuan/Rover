# Rover

Rover is a general-purpose inference engine for running models locally, with support for multimodal inputs and resumable execution.

## Features

- **Text and image inputs** — generate responses from prompts and images through the command line or Python API.
- **Resumable sessions** — save inference progress and continue a paused or interrupted request.
- **Chunked prefill** — process prompts in configurable chunks.
- **Configurable execution** — choose KV cache formats, scheduling policies, and CUDA graph execution.

The current model backend supports Qwen3-VL. Model weights are loaded from a local directory.

## Installation

Rover requires Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/). A CUDA-capable GPU is recommended; CPU execution is available with the default dense cache.

```bash
git clone https://github.com/OPEN-TianSuan/Rover.git
cd Rover
uv sync
```

Tokenization and image preprocessing are included in the default installation.

### NVIDIA Jetson

On Jetson Orin with JetPack 6 and CUDA 12.6, replace `uv sync` with:

```bash
python3 scripts/setup_jetson.py
```

This installs the compatible device dependencies and prepares the environment.

## Quick start

Download a supported model, such as Qwen3-VL-2B-Instruct, and replace `/path/to/model` with its local directory. The directory must contain the model weights, configuration, and tokenizer files.

```bash
uv run rover generate \
  --model /path/to/model \
  --prompt 'Explain why the sky is blue in two sentences.' \
  --state-dir runs/example \
  --device cuda \
  --max-new-tokens 128
```

Rover prints the response and saves the session in `runs/example`. Use a new, empty session directory for each request. To run on CPU, replace `--device cuda` with `--device cpu`.

### Image inputs

```bash
uv run rover generate \
  --model /path/to/model \
  --prompt 'Describe this image.' \
  --image /path/to/photo.jpg \
  --state-dir runs/image \
  --device cuda \
  --max-new-tokens 128
```

Repeat `--image` to attach multiple images.

### Pause and resume

Use `--stop-after-tokens` to pause generation after a specified number of output tokens:

```bash
uv run rover generate \
  --model /path/to/model \
  --prompt 'Write a short story about a lost robot.' \
  --state-dir runs/story \
  --device cuda \
  --max-new-tokens 128 \
  --stop-after-tokens 16
```

Continue the saved session with:

```bash
uv run rover resume --state-dir runs/story
```

Resume uses the saved model path and settings and continues up to the original output limit or the model's end-of-sequence token. Keep the model directory available, and do not run multiple processes against the same session directory.

## Python API

```python
from rover.engine import Engine
from rover.processing import Processor

model_dir = '/path/to/model'
processor = Processor(model_dir)
inputs = processor.encode('Introduce yourself in one sentence.')

with Engine.create(
    model_dir,
    'runs/python-example',
    device='cuda',
    max_new_tokens=128,
    **inputs,
) as engine:
    tokens = engine.generate()
    print(processor.decode(tokens))
```

For image inputs, pass image paths as the second argument:

```python
inputs = processor.encode('Describe this image.', ['/path/to/photo.jpg'])
```

## Configuration

Run `uv run rover generate --help` for the full list of options.

| Option | Description |
| --- | --- |
| `--max-new-tokens` | Maximum number of generated tokens. |
| `--chunk-size` | Number of prompt tokens processed per prefill chunk. |
| `--prefill-order` | Prefill execution order: `chunk_first`, `layer_first`, or `scheduled`. |
| `--strategy` | Scheduling policy when using scheduled prefill. |
| `--cache-format` | KV cache format: `dense`, `kivi`, or `turboquant`. |
| `--cuda-graph` | Enable CUDA graphs for supported decode paths. |
| `--runtime-options` | Path to a JSON file containing additional runtime settings. |

The default dense cache does not require a custom extension. KIVI and TurboQuant require CUDA kernels, which can be built with:

```bash
uv run python scripts/build_kernels.py
```

On Jetson Orin, you can explicitly select the GPU architecture and limit compilation parallelism:

```bash
TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=2 uv run python scripts/build_kernels.py
```

## Current limitations

- One request at a time, with batch size 1.
- The CLI supports text and still images; video file decoding is not available.
- Recovery starts from committed steps and is not guaranteed at every hardware power-loss boundary.
- Asynchronous TurboQuant compression may switch cache formats at a different point after a pause, affecting subsequent output. Use dense or synchronous cache modes when repeatable recovery output is required.
- GPTQ weights require a separately installed, compatible GPTQ runtime. The examples above use non-GPTQ weights.

## Development

```bash
uv run pytest -q
uv run ruff check rover tests scripts
uv build
```
