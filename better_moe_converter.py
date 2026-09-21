#!/usr/bin/env python3
"""One-command dense Qwen3.5 to quality-gated routed MoE conversion."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "qwen_moe.py"


def run(command):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def copy_overlay(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob("layer_*.safetensors"):
        shutil.copy2(path, destination / path.name)
    for name in ("recipe.json", "report.json", "alignment_metrics.json"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


def restore_overlay(backup: Path, output: Path):
    for path in backup.glob("layer_*.safetensors"):
        shutil.copy2(path, output / path.name)
    for name in ("recipe.json", "report.json", "alignment_metrics.json"):
        path = backup / name
        if path.is_file():
            shutil.copy2(path, output / name)


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Convert a dense Qwen3.5 checkpoint to an eight-expert, top-k-1 "
                     "shared-backbone MoE and accept it only when held-out loss is no worse."))
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output", default="qwen35_2b/better_moe")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--train-blocks", type=int, default=128)
    parser.add_argument("--validation-blocks", type=int, default=8)
    parser.add_argument("--test-blocks", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1 or args.adapter_rank < 1:
        raise SystemExit("--steps and --adapter-rank must be positive")
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    recipe_path = output / "recipe.json"
    summary_path = output / "conversion_summary.json"

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        print(json.dumps(summary, indent=2))
        print("Conversion is already complete. Choose another --output for a new recipe.")
        return

    if not recipe_path.is_file():
        run([
            sys.executable, ENGINE, "convert", "--model", args.model,
            "--output", output, "--device", args.device,
            "--experts", 8, "--top-k", 1, "--adapter-rank", args.adapter_rank,
            "--shared-backbone", "--steps", 0,
            "--train-blocks", 32, "--validation-blocks", args.validation_blocks,
            "--test-blocks", args.test_blocks, "--sequence-length", args.sequence_length,
            "--seed", args.seed,
        ])
    recipe = json.loads(recipe_path.read_text())
    if recipe.get("mode") != "shared_backbone" or recipe.get("experts") != 8:
        raise SystemExit(f"{output} is not an eight-expert shared-backbone checkpoint")

    backup = output / "baseline_overlay"
    if not backup.is_dir():
        copy_overlay(output, backup)

    try:
        run([
            sys.executable, ENGINE, "recover", "--checkpoint", output,
            "--device", args.device, "--steps", args.steps,
            "--learning-rate", args.learning_rate,
            "--train-blocks", args.train_blocks,
            "--validation-blocks", args.validation_blocks,
            "--test-blocks", args.test_blocks,
            "--sequence-length", args.sequence_length,
            "--eval-every", args.eval_every, "--seed", args.seed,
        ])
    except Exception:
        restore_overlay(backup, output)
        raise
    recovery = json.loads((output / "recovery_report.json").read_text())
    dense_nll = recovery["dense"]["nll"]
    candidate_nll = recovery["moe"]["nll"]
    accepted = candidate_nll <= dense_nll
    if not accepted:
        restore_overlay(backup, output)

    summary = {
        "status": "accepted" if accepted else "rejected_and_restored_dense_equivalent",
        "model": args.model,
        "architecture": "frozen shared dense FFN plus one of eight routed expert adapters",
        "experts_per_layer": 8,
        "active_experts_per_token_per_layer": 1,
        "dense_test_nll": dense_nll,
        "candidate_test_nll": candidate_nll,
        "dense_test_perplexity": recovery["dense"]["perplexity"],
        "candidate_test_perplexity": recovery["moe"]["perplexity"],
        "quality_gate": "candidate held-out NLL <= dense held-out NLL",
        "checkpoint": str(output),
        "scope": ("Acceptance applies to this held-out text split; it does not prove "
                  "better general reasoning, knowledge, or vision capability."),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if accepted:
        print(f"Accepted MoE checkpoint: {output}")
    else:
        print("Candidate missed the quality gate; restored the dense-equivalent MoE overlay.")


if __name__ == "__main__":
    main()
