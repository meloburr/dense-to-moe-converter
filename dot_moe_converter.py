#!/usr/bin/env python3
"""Paper-faithful front end for dense-to-DOT-MoE conversion.

The converter derives the expert count from the source FFN width, then invokes
the differentiable optimal-transport alignment and exports a compact overlay.
The original Hugging Face checkpoint remains unchanged.
"""

import argparse
import json
import math
from pathlib import Path

from transformers import AutoConfig

from qwen_moe import convert, quiet_library_output


PAPER_MODEL = "Qwen/Qwen2.5-7B"


def model_dimensions(model_id: str) -> tuple[int, int]:
    config = AutoConfig.from_pretrained(model_id)
    config = getattr(config, "text_config", config)
    return int(config.hidden_size), int(config.intermediate_size)


def derive_topology(intermediate_size: int, expert_size: int,
                    active_fraction: float) -> tuple[int, int]:
    if expert_size < 1 or intermediate_size % expert_size:
        raise ValueError(
            f"FFN width {intermediate_size} is not divisible by expert size {expert_size}")
    if not 0 < active_fraction <= 1:
        raise ValueError("--active-fraction must be in (0, 1]")
    experts = intermediate_size // expert_size
    top_k = max(1, min(experts, math.floor(experts * active_fraction + 0.5)))
    return experts, top_k


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a dense LLaMA/Qwen-style checkpoint with DOT-MoE.")
    parser.add_argument("--model", default=PAPER_MODEL)
    parser.add_argument("--output", default="outputs/qwen2.5-7b-dot-moe")
    parser.add_argument("--profile", choices=("paper", "smoke"), default="smoke",
                        help="Paper settings need 8xH100-class hardware; smoke validates the pipeline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"),
                        default="bfloat16")
    parser.add_argument("--expert-size", type=int, default=128)
    parser.add_argument("--active-fraction", type=float, default=0.25)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--train-blocks", type=int)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline-teacher", action="store_true",
                        help="Cache teacher logits (faster, but can require very large host RAM)")
    return parser.parse_args()


def main():
    cli = parse_args()
    quiet_library_output()
    hidden_size, intermediate_size = model_dimensions(cli.model)
    experts, top_k = derive_topology(
        intermediate_size, cli.expert_size, cli.active_fraction)

    paper = cli.profile == "paper"
    steps = cli.steps if cli.steps is not None else (3500 if paper else 10)
    batch_size = cli.batch_size if cli.batch_size is not None else (64 if paper else 1)
    sequence_length = cli.sequence_length if cli.sequence_length is not None else (2048 if paper else 128)
    train_blocks = cli.train_blocks if cli.train_blocks is not None else (batch_size * 8 if paper else 8)

    if cli.device == "cpu" and cli.dtype != "float32":
        raise ValueError("Use --dtype float32 for the CPU smoke profile")

    settings = {
        "command": "convert",
        "model": cli.model,
        "output": str(Path(cli.output)),
        "device": cli.device,
        "dtype": cli.dtype,
        "experts": experts,
        "top_k": top_k,
        "steps": steps,
        "learning_rate": cli.learning_rate,
        "batch_tokens": 64,
        "batch_size": batch_size,
        "train_blocks": train_blocks,
        "validation_blocks": 4 if not paper else 32,
        "test_blocks": 8 if not paper else 32,
        "sequence_length": sequence_length,
        "seed": cli.seed,
        "adapter_rank": 0,
        "shared_backbone": False,
        "global_alignment": True,
        "kl_weight": 2.0,
        "ce_weight": 1.0,
        "z_loss_weight": 1e-3,
        "balance_weight": 1e-2,
        "resume": cli.resume,
        "online_teacher": not cli.offline_teacher,
        "train_dataset": "allenai/dolmino-mix-1124" if paper else "Salesforce/wikitext",
        "train_dataset_config": "default" if paper else "wikitext-2-raw-v1",
        "train_split": "train",
        "stream_train": paper,
    }
    summary = {
        "model": cli.model,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "expert_size": cli.expert_size,
        "experts": experts,
        "active_experts": top_k,
        "active_ffn_fraction": top_k / experts,
        "profile": cli.profile,
        "warning": ("The paper profile mirrors reported hyperparameters and is expensive."
                    if paper else "Smoke profile validates conversion mechanics, not paper metrics."),
    }
    print(json.dumps(summary, indent=2), flush=True)
    convert(argparse.Namespace(**settings))


if __name__ == "__main__":
    main()
