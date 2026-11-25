# Rover

[English](README.md) | [简体中文](README.zh-CN.md) | [架构文档](https://github.com/OPEN-TianSuan/Rover/blob/main/doc/architecture.md "architecture.md")

Rover 是一款通用的本地模型推理引擎，支持多模态输入和断点恢复。

## 功能特性

- **文本和图像输入**：通过命令行或 Python API 使用文本与图像生成响应。
- **可恢复会话**：保存推理进度，并继续已暂停或中断的请求。
- **分块预填充**：按可配置的块处理提示词。
- **灵活的执行配置**：可选择 KV 缓存格式、调度策略和 CUDA Graph 执行。

当前模型后端支持 Qwen3-VL。模型权重从本地目录加载。

## 文档

系统设计和实现细节请参阅[架构文档](https://github.com/OPEN-TianSuan/Rover/blob/main/doc/architecture.md "architecture.md")。

## 安装

Rover 需要 Python 3.10–3.13 和 [uv](https://docs.astral.sh/uv/)。推荐使用支持 CUDA 的 GPU；使用默认的稠密缓存时也可以在 CPU 上运行。

```bash
git clone https://github.com/OPEN-TianSuan/Rover.git
cd Rover
uv sync
```

默认安装已包含分词和图像预处理所需的依赖。

### NVIDIA Jetson

在搭载 JetPack 6 和 CUDA 12.6 的 Jetson Orin 上，请使用以下命令代替 `uv sync`：

```bash
python3 scripts/setup_jetson.py
```

该脚本会安装兼容的设备依赖并准备运行环境。

## 快速开始

下载受支持的模型，例如 Qwen3-VL-2B-Instruct，并将 `/path/to/model` 替换为模型本地目录。该目录必须包含模型权重、配置和分词器文件。

```bash
uv run rover generate \
  --model /path/to/model \
  --prompt '请用两句话解释天空为什么是蓝色的。' \
  --state-dir runs/example \
  --device cuda \
  --max-new-tokens 128
```

Rover 会输出生成结果，并将会话保存在 `runs/example`。每个请求请使用一个新的空会话目录。在 CPU 上运行时，将 `--device cuda` 替换为 `--device cpu`。

### 图像输入

```bash
uv run rover generate \
  --model /path/to/model \
  --prompt '描述这张图片。' \
  --image /path/to/photo.jpg \
  --state-dir runs/image \
  --device cuda \
  --max-new-tokens 128
```

可以重复传入 `--image` 以附加多张图片。

### 暂停与恢复

使用 `--stop-after-tokens` 可以在生成指定数量的词元后暂停：

```bash
uv run rover generate \
  --model /path/to/model \
  --prompt '写一个关于迷路机器人的短篇故事。' \
  --state-dir runs/story \
  --device cuda \
  --max-new-tokens 128 \
  --stop-after-tokens 16
```

使用以下命令继续已保存的会话：

```bash
uv run rover resume --state-dir runs/story
```

恢复时会使用已保存的模型路径和设置，并继续生成，直到达到原定输出上限或模型生成结束标记。请确保模型目录仍可用，并避免多个进程同时访问同一个会话目录。

## Python API

```python
from rover.engine import Engine
from rover.processing import Processor

model_dir = '/path/to/model'
processor = Processor(model_dir)
inputs = processor.encode('请用一句话介绍你自己。')

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

输入图像时，将图像路径作为第二个参数传入：

```python
inputs = processor.encode('描述这张图片。', ['/path/to/photo.jpg'])
```

## 配置

运行 `uv run rover generate --help` 查看全部选项。

| 选项 | 说明 |
| --- | --- |
| `--max-new-tokens` | 最多生成的词元数量。 |
| `--chunk-size` | 每个预填充块包含的提示词词元数量。 |
| `--prefill-order` | 预填充执行顺序：`chunk_first`、`layer_first` 或 `scheduled`。 |
| `--strategy` | 使用 scheduled 预填充时的调度策略。 |
| `--cache-format` | KV 缓存格式：`dense`、`kivi` 或 `turboquant`。 |
| `--cuda-graph` | 为受支持的解码路径启用 CUDA Graph。 |
| `--runtime-options` | 附加运行时设置 JSON 文件的路径。 |

默认的稠密缓存无需自定义扩展。KIVI 和 TurboQuant 需要 CUDA 内核，可通过以下命令构建：

```bash
uv run python scripts/build_kernels.py
```

在 Jetson Orin 上，可以显式指定 GPU 架构并限制编译并行数：

```bash
TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=2 uv run python scripts/build_kernels.py
```

## 当前限制

- 一次只能处理一个请求，批量大小为 1。
- CLI 支持文本和静态图像输入，暂不支持视频文件解码。
- 故障恢复从已提交的步骤继续，无法保证在硬件断电的任意时刻恢复。
- TurboQuant 异步压缩可能在不同时间切换缓存格式；暂停后继续运行时，切换时机可能变化并影响后续输出。需要可重复的恢复结果时，请使用稠密缓存或同步缓存模式。
- GPTQ 权重需要单独安装兼容的 GPTQ 运行时。以上示例使用非 GPTQ 权重。

## 开发

```bash
uv run pytest -q
uv run ruff check rover tests scripts
uv build
```
