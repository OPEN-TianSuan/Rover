"""Command-line interface. Model weights and processors are always local."""

import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(prog="rover", description="A general-purpose inference engine with resumable sessions")
    commands = p.add_subparsers(dest="command", required=True)
    run = commands.add_parser("generate", help="Start a new inference session")
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--state-dir", required=True, type=Path)
    inputs = run.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt")
    inputs.add_argument("--input-ids", type=Path, help="Read pre-tokenized input from a JSON array")
    run.add_argument("--image", action="append", type=Path, default=[])
    run.add_argument("--device", default="cpu")
    run.add_argument("--max-new-tokens", type=int, default=128)
    run.add_argument("--chunk-size", type=int, default=256)
    run.add_argument("--prefill-order", choices=["chunk_first", "layer_first", "scheduled"], default="chunk_first")
    run.add_argument(
        "--strategy", choices=["chunk_first", "layer_first", "greedy", "state_compression_dp"], default="chunk_first"
    )
    run.add_argument("--cache-format", choices=["dense", "kivi", "turboquant"], default="dense")
    run.add_argument("--cuda-graph", action="store_true")
    run.add_argument("--stop-after-tokens", type=int, help="Pause once this many output tokens have been checkpointed")
    run.add_argument("--runtime-options", type=Path, help="JSON RuntimeConfig overrides (e.g. quantization/profiling)")
    resume = commands.add_parser("resume", help="Continue a saved session using its runtime settings")
    resume.add_argument("--state-dir", required=True, type=Path)
    info = commands.add_parser("inspect", help="Read checkpoint metadata without loading weights")
    info.add_argument("--state-dir", required=True, type=Path)
    return p


def prepare_inputs(args):
    import torch

    if args.input_ids:
        if args.image:
            raise ValueError("--image requires --prompt")
        values = json.loads(args.input_ids.read_text())
        return {"input_ids": torch.tensor([values], dtype=torch.long)}, None
    from .processing import Processor

    processor = Processor(args.model)
    return processor.encode(args.prompt, args.image), processor


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == "inspect":
        print((args.state_dir / "metadata.json").read_text())
        return
    from .engine import Engine

    tokenizer = None
    if args.command == "resume":
        engine = Engine.resume(args.state_dir)
        try:
            from .processing import Processor

            tokenizer = Processor(engine.state.runtime_config.model_dir)
        except (ImportError, OSError):
            pass
    else:
        inputs, tokenizer = prepare_inputs(args)
        inputs.pop("attention_mask", None)
        inputs.pop("token_type_ids", None)
        options = json.loads(args.runtime_options.read_text()) if args.runtime_options else None
        engine = Engine.create(
            args.model,
            args.state_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            chunk_size=args.chunk_size,
            prefill_order=args.prefill_order,
            strategy=args.strategy,
            cache_format=args.cache_format,
            cuda_graph=args.cuda_graph,
            stop_after_tokens=args.stop_after_tokens,
            runtime_options=options,
            **inputs,
        )
    with engine:
        from rover.diagnostics.tracing import tracer

        if engine.state.runtime_config.trace_enabled:
            tracer.enable(str(Path(engine.state.runtime_config.state_dir) / "trace.jsonl"))
        tokens = engine.generate()
        tracer.flush()
        print(tokenizer.decode(tokens, skip_special_tokens=True) if tokenizer else json.dumps(tokens))
        result = {"token_ids": tokens, "phase_timings_ms": engine.model.last_phase_timings_ms}
        (Path(engine.state.runtime_config.state_dir) / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
