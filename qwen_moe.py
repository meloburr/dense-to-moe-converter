"""Calibrate and load a Qwen3.5 dense-to-MoE research checkpoint.

The saved overlay contains expert membership and routers. Original frozen weights
come from the pinned Hugging Face checkpoint. This is a custom PyTorch runtime.
"""
import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.utils import logging as transformers_logging
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors.torch import load_file, save_file

from ot_moe import AlignmentFFN, SharedBackboneMoE, SparseGatedMoE, balanced_round, sinkhorn

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def quiet_library_output():
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()


def load_dense(model_id, revision=None, device="cpu", dtype="float32"):
    patterns = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"]
    try:
        path = snapshot_download(model_id, revision=revision, allow_patterns=patterns,
                                 local_files_only=True)
    except LocalEntryNotFoundError:
        path = snapshot_download(model_id, revision=revision,
                                 allow_patterns=patterns, max_workers=2)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {dtype!r}")
    config = AutoConfig.from_pretrained(path)
    # Qwen3.5 is multimodal and is not registered as AutoModelForCausalLM.
    # The paper's LLaMA/Qwen2.5 checkpoints use the generic causal-LM path.
    model_class = (Qwen3_5ForConditionalGeneration
                   if config.model_type == "qwen3_5" else AutoModelForCausalLM)
    model = model_class.from_pretrained(
        path, dtype=dtype_map[dtype], attn_implementation="eager").eval().to(device)
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer, Path(path).name


def text_layers(model):
    backbone = model.model
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    if not hasattr(backbone, "layers"):
        raise ValueError(f"Unsupported transformer layout: {type(model).__name__}")
    return backbone.layers


def blocks_from_dataset(tokenizer, split, count, length,
                        dataset_id="Salesforce/wikitext",
                        dataset_config="wikitext-2-raw-v1", streaming=False):
    from datasets import load_dataset
    dataset = load_dataset(dataset_id, dataset_config or None, split=split,
                           streaming=streaming)
    ids = []
    texts = (row["text"] for row in dataset) if streaming else dataset["text"]
    for text in texts:
        if text.strip():
            ids.extend(tokenizer.encode(text + "\n", add_special_tokens=False))
        if len(ids) >= count * length:
            break
    if len(ids) < count * length:
        raise ValueError("Insufficient calibration text")
    return torch.tensor(ids[:count * length], dtype=torch.long).reshape(count, length)


@torch.no_grad()
def collect_inputs(model, blocks, device):
    captured = [[] for _ in text_layers(model)]
    def capture(index):
        def hook(module, args):
            captured[index].append(args[0].detach().reshape(-1, args[0].shape[-1]).cpu())
        return hook
    hooks = [layer.mlp.register_forward_pre_hook(capture(i))
             for i, layer in enumerate(text_layers(model))]
    try:
        for block in blocks:
            model(input_ids=block[None].to(device), use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    return [torch.cat(parts) for parts in captured]


@torch.no_grad()
def evaluate(model, blocks, device):
    total_loss, count = 0.0, 0
    start = time.perf_counter()
    for block in blocks:
        ids = block[None].to(device)
        logits = model(input_ids=ids, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1), reduction="sum")
        total_loss += loss.item()
        count += ids.shape[1] - 1
    nll = total_loss / count
    return {"nll": nll, "perplexity": math.exp(min(nll, 80)), "tokens": count,
            "elapsed_seconds": time.perf_counter() - start}


@torch.no_grad()
def example_generation(model, tokenizer, device):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain why the sky is blue in one sentence."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def train_layer(dense, train, validation, args, device):
    alignment = AlignmentFFN(dense, args.experts, args.top_k, args.adapter_rank).to(device)
    trainable = [parameter for name, parameter in alignment.named_parameters()
                 if not name.startswith("dense.")]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    validation = validation[:256].to(device)
    with torch.no_grad():
        target_validation = dense(validation)
        scale = target_validation.float().square().mean().clamp_min(1e-8)
    best_loss, best_state, initial_loss = float("inf"), None, None
    for step in range(args.steps + 1):
        if step % 20 == 0 or step == args.steps:
            with torch.no_grad():
                pred = alignment(validation, 0.1)[0]
                error = (F.mse_loss(pred.float(), target_validation.float()) / scale).item()
                if initial_loss is None:
                    initial_loss = error
                if error < best_loss:
                    best_loss = error
                    best_state = {"affinity": alignment.affinity.detach().cpu().clone(),
                                  "router": alignment.router.weight.detach().cpu().clone()}
                    if alignment.residual_up is not None:
                        best_state["residual_up"] = alignment.residual_up.weight.detach().cpu().clone()
                        best_state["residual_down"] = alignment.residual_down.weight.detach().cpu().clone()
        if step == args.steps:
            break
        rows = torch.randint(len(train), (args.batch_tokens,))
        x = train[rows].to(device)
        with torch.no_grad():
            target = dense(x)
        temperature = max(0.1, 1.0 - 0.9 * step / max(args.steps - 1, 1))
        prediction, balance, z_loss = alignment(x, temperature)
        reconstruction = F.mse_loss(prediction.float(), target.float()) / scale
        loss = reconstruction + 0.001 * balance + 0.00001 * z_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite alignment loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    with torch.no_grad():
        alignment.affinity.copy_(best_state["affinity"].to(device))
        alignment.router.weight.copy_(best_state["router"].to(device))
        if alignment.residual_up is not None:
            alignment.residual_up.weight.copy_(best_state["residual_up"].to(device))
            alignment.residual_down.weight.copy_(best_state["residual_down"].to(device))
        partition = balanced_round(sinkhorn(alignment.affinity, 0.1))
        sparse = SparseGatedMoE(dense, alignment.router, partition, args.top_k,
                                alignment.residual_up, alignment.residual_down).eval()
        sample = validation[:16]
        masked = alignment(sample, 0.1)[0]
        exported = sparse(sample)
        export_error = (masked - exported).abs().max().item()
        torch.testing.assert_close(masked, exported, atol=2e-5, rtol=2e-4)
        sparse.top_k = args.experts
        torch.testing.assert_close(sparse(sample), dense(sample), atol=2e-5, rtol=2e-4)
        sparse.top_k = args.top_k
    tensors = {"membership": partition.argmax(dim=1).cpu().contiguous(),
               "router": alignment.router.weight.detach().cpu().contiguous()}
    if alignment.residual_up is not None:
        tensors["residual_up"] = alignment.residual_up.weight.detach().cpu().contiguous()
        tensors["residual_down"] = alignment.residual_down.weight.detach().cpu().contiguous()
    metrics = {"initial_validation_relative_mse": initial_loss,
               "validation_relative_mse": best_loss,
               "export_max_absolute_error": export_error}
    return tensors, metrics


@torch.no_grad()
def balanced_token_router(hidden_states, experts, iterations=20):
    """Fit balanced spherical centroids with Sinkhorn assignments."""
    x = F.normalize(hidden_states.float(), dim=-1)
    initial = torch.linspace(0, len(x) - 1, experts, device=x.device).long()
    centroids = x.index_select(0, initial).clone()
    assignment = None
    for _ in range(iterations):
        plan = sinkhorn(x @ centroids.T, temperature=0.1)
        assignment = balanced_round(plan)
        centroids = assignment.T @ x
        centroids = F.normalize(centroids, dim=-1)
    counts = assignment.sum(0)
    return centroids.cpu().contiguous(), counts.cpu().tolist()


def global_dot_alignment(model, train_blocks, teacher_logits, args, device):
    """Joint DOT-MoE alignment with the paper's four-term objective."""
    alignments = []
    for layer in text_layers(model):
        alignment = AlignmentFFN(layer.mlp, args.experts, args.top_k, 0).to(device)
        layer.mlp = alignment
        alignments.append(alignment)
    trainable = [parameter for alignment in alignments
                 for name, parameter in alignment.named_parameters()
                 if not name.startswith("dense.")]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=1e-4)
    warmup_steps = max(1, int(args.steps * 0.2))
    state_file = Path(args.output) / "training_state.safetensors"
    progress_file = Path(args.output) / "training_progress.json"
    start_step = 0
    if args.resume and state_file.is_file() and progress_file.is_file():
        state = load_file(str(state_file))
        for index, alignment in enumerate(alignments):
            alignment.affinity.data.copy_(state[f"affinity_{index:02d}"].to(device))
            alignment.router.weight.data.copy_(state[f"router_{index:02d}"].to(device))
        start_step = json.loads(progress_file.read_text())["completed_steps"]
        print(f"Resuming global alignment at step {start_step + 1}", flush=True)
    def lr_multiplier(current_step):
        if current_step < warmup_steps:
            return (current_step + 1) / warmup_steps
        progress = (current_step - warmup_steps) / max(args.steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    for step in range(start_step, args.steps):
        learning_rate = args.learning_rate * lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        start = (step * args.batch_size) % len(train_blocks)
        indices = [(start + offset) % len(train_blocks)
                   for offset in range(args.batch_size)]
        ids = train_blocks[indices].to(device)
        temperature = max(0.1, 1.0 - 0.9 * min(step / warmup_steps, 1.0))
        for alignment in alignments:
            alignment.temperature = temperature
        if args.online_teacher:
            for alignment in alignments:
                alignment.top_k = args.experts
            with torch.no_grad():
                dense_logits = model(input_ids=ids, use_cache=False).logits.float()
            for alignment in alignments:
                alignment.top_k = args.top_k
        else:
            dense_logits = torch.cat(
                [teacher_logits[index] for index in indices], dim=0).to(device).float()
        logits = model(input_ids=ids, use_cache=False).logits
        student_log_probs = F.log_softmax(logits.float(), dim=-1)
        teacher_probs = F.softmax(dense_logits, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="sum") / ids.numel()
        ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                             ids[:, 1:].reshape(-1))
        z_loss = torch.stack([alignment.last_z_loss for alignment in alignments]).mean()
        balance = torch.stack([alignment.last_balance for alignment in alignments]).mean()
        loss = (args.kl_weight * kl + args.ce_weight * ce
                + args.z_loss_weight * z_loss + args.balance_weight * balance)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite global alignment loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step % 10 == 0 or step + 1 == args.steps:
            print(json.dumps({"step": step + 1, "loss": loss.item(), "kl": kl.item(),
                              "ce": ce.item(), "balance": balance.item(),
                              "z_loss": z_loss.item(), "temperature": temperature,
                              "learning_rate": learning_rate}), flush=True)
        if (step + 1) % 25 == 0:
            checkpoint = {}
            for index, alignment in enumerate(alignments):
                checkpoint[f"affinity_{index:02d}"] = alignment.affinity.detach().cpu().contiguous()
                checkpoint[f"router_{index:02d}"] = alignment.router.weight.detach().cpu().contiguous()
            save_file(checkpoint, str(state_file))
            progress_file.write_text(json.dumps({"completed_steps": step + 1}, indent=2))
    layer_metrics = []
    for index, alignment in enumerate(alignments):
        with torch.no_grad():
            partition = balanced_round(sinkhorn(alignment.affinity, 0.1))
        tensors = {"membership": partition.argmax(dim=1).cpu().contiguous(),
                   "router": alignment.router.weight.detach().cpu().contiguous()}
        save_file(tensors, str(Path(args.output) / f"layer_{index:02d}.safetensors"))
        layer_metrics.append({"layer": index,
                              "expert_neuron_counts": partition.sum(0).cpu().tolist()})
    for index, alignment in enumerate(alignments):
        text_layers(model)[index].mlp = SparseGatedMoE(
            alignment.dense, alignment.router,
            balanced_round(sinkhorn(alignment.affinity, 0.1)), args.top_k).eval()
    return layer_metrics


def install_overlay(model, folder, top_k=None, allow_untrained_top_k=False):
    folder = Path(folder)
    metadata = json.loads((folder / "recipe.json").read_text())
    experts = metadata["experts"]
    k = metadata["top_k"] if top_k is None else top_k
    if not 1 <= k <= experts:
        raise ValueError("Invalid active expert count")
    if k not in (metadata["top_k"], experts) and not allow_untrained_top_k:
        raise ValueError(
            f"This router was trained for top-k {metadata['top_k']}. "
            f"Use top-k {metadata['top_k']} or {experts}; choose a checkpoint trained "
            "for another setting instead of overriding it.")
    for index in metadata["layers"]:
        dense = text_layers(model)[index].mlp
        tensors = load_file(str(folder / f"layer_{index:02d}.safetensors"))
        device = dense.gate_proj.weight.device
        if metadata.get("mode") == "shared_backbone":
            text_layers(model)[index].mlp = SharedBackboneMoE(
                dense, tensors["router"].to(device), tensors["adapter_up"].to(device),
                tensors["adapter_down"].to(device), k).eval()
            continue
        partition = F.one_hot(tensors["membership"], experts).float().to(device)
        if not bool((partition.sum(0) == partition.shape[0] // experts).all()):
            raise ValueError("Unbalanced checkpoint")
        router = nn.Linear(
            dense.gate_proj.in_features, experts, bias=False, device=device,
            dtype=dense.gate_proj.weight.dtype)
        router.weight.data.copy_(tensors["router"].to(device))
        residual_up = residual_down = None
        if "residual_up" in tensors:
            rank = tensors["residual_up"].shape[0]
            residual_up = nn.Linear(dense.gate_proj.in_features, rank, bias=False, device=device)
            residual_down = nn.Linear(rank, dense.down_proj.out_features, bias=False, device=device)
            residual_up.weight.data.copy_(tensors["residual_up"].to(device))
            residual_down.weight.data.copy_(tensors["residual_down"].to(device))
        text_layers(model)[index].mlp = SparseGatedMoE(
            dense, router, partition, k, residual_up, residual_down).eval()
    model.requires_grad_(False)
    return model


def recover_shared_backbone(args):
    """Fine-tune routed expert adapters and keep the best validation checkpoint."""
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    folder = Path(args.checkpoint)
    metadata = json.loads((folder / "recipe.json").read_text())
    if metadata.get("mode") != "shared_backbone":
        raise ValueError("Recovery requires a --shared-backbone checkpoint")
    if metadata.get("top_k") != 1 or metadata.get("experts") != 8:
        raise ValueError("This recovery recipe expects eight experts with top-k 1")
    model, tokenizer, _ = load_dense(
        metadata["model_id"], metadata.get("revision"), args.device,
        metadata.get("settings", {}).get("dtype", "float32"))
    install_overlay(model, folder)
    train = blocks_from_dataset(tokenizer, "train", args.train_blocks, args.sequence_length)
    validation = blocks_from_dataset(
        tokenizer, "validation", args.validation_blocks, args.sequence_length)
    test = blocks_from_dataset(tokenizer, "test", args.test_blocks, args.sequence_length)
    baseline_validation = evaluate(model, validation, args.device)
    baseline_test = evaluate(model, test, args.device)
    print(json.dumps({"dense_validation": baseline_validation,
                      "dense_test": baseline_test}), flush=True)

    modules = [text_layers(model)[index].mlp for index in metadata["layers"]]
    trainable = []
    for module in modules:
        if not isinstance(module, SharedBackboneMoE):
            raise ValueError("Checkpoint did not install shared-backbone expert modules")
        # A zero down projection preserves the dense model.  A nonzero up
        # projection is required so gradients can reach the down projection.
        if not bool(module.adapter_up.detach().abs().max()):
            nn.init.normal_(module.adapter_up, mean=0.0, std=0.02)
        module.adapter_up.requires_grad_(True)
        module.adapter_down.requires_grad_(True)
        trainable.extend((module.adapter_up, module.adapter_down))

    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    best_nll = baseline_validation["nll"]
    best_step = 0
    best_state = [(module.adapter_up.detach().cpu().clone(),
                   module.adapter_down.detach().cpu().clone())
                  for module in modules]
    model.train()
    for step in range(args.steps):
        ids = train[step % len(train):step % len(train) + 1].to(args.device)
        logits = model(input_ids=ids, use_cache=False).logits
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite recovery loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.steps:
            model.eval()
            validation_result = evaluate(model, validation, args.device)
            improved = validation_result["nll"] < best_nll
            if improved:
                best_nll = validation_result["nll"]
                best_step = completed
                best_state = [(module.adapter_up.detach().cpu().clone(),
                               module.adapter_down.detach().cpu().clone())
                              for module in modules]
            print(json.dumps({"step": completed, "train_loss": loss.item(),
                              "validation_nll": validation_result["nll"],
                              "validation_perplexity": validation_result["perplexity"],
                              "best_step": best_step}), flush=True)
            model.train()

    for module, (adapter_up, adapter_down) in zip(modules, best_state):
        module.adapter_up.data.copy_(adapter_up.to(args.device))
        module.adapter_down.data.copy_(adapter_down.to(args.device))
        module.adapter_up.requires_grad_(False)
        module.adapter_down.requires_grad_(False)
    model.eval()
    for index, module in zip(metadata["layers"], modules):
        tensors = load_file(str(folder / f"layer_{index:02d}.safetensors"))
        tensors["adapter_up"] = module.adapter_up.detach().cpu().contiguous()
        tensors["adapter_down"] = module.adapter_down.detach().cpu().contiguous()
        save_file(tensors, str(folder / f"layer_{index:02d}.safetensors"))

    recovered_test = evaluate(model, test, args.device)
    report = {
        "dense": baseline_test,
        "moe": recovered_test,
        "dense_validation": baseline_validation,
        "best_validation_nll": best_nll,
        "best_step": best_step,
        "moe_generation": example_generation(model, tokenizer, args.device),
        "experts": 8,
        "top_k": 1,
        "architecture": "shared dense FFN plus one selected expert adapter",
        "claim": ("better_on_this_test_split" if recovered_test["nll"] < baseline_test["nll"]
                  else "not_better_on_this_test_split"),
    }
    (folder / "recovery_report.json").write_text(json.dumps(report, indent=2))
    metadata["recovery"] = {
        "steps": args.steps, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "sequence_length": args.sequence_length,
        "train_blocks": args.train_blocks, "validation_blocks": args.validation_blocks,
        "test_blocks": args.test_blocks, "best_step": best_step,
    }
    metadata["status"] = "recovered_and_evaluated"
    (folder / "recipe.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def convert(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    recipe = folder / "recipe.json"
    if recipe.exists():
        existing = json.loads(recipe.read_text())
        matching = existing.get("model_id") == args.model and all(
            existing.get("settings", {}).get(key, 0 if key == "adapter_rank" else None)
            == getattr(args, key)
            for key in ("device", "experts", "top_k", "steps", "learning_rate",
                        "batch_tokens", "train_blocks", "validation_blocks",
                        "test_blocks", "sequence_length", "seed", "adapter_rank",
                        "shared_backbone", "global_alignment", "kl_weight", "ce_weight",
                        "z_loss_weight", "balance_weight", "online_teacher",
                        "batch_size", "dtype", "train_dataset",
                        "train_dataset_config", "train_split", "stream_train"))
        complete = (existing.get("status") == "converted_and_smoke_evaluated"
                    and (folder / "report.json").is_file()
                    and all((folder / f"layer_{index:02d}.safetensors").is_file()
                            for index in existing.get("layers", [])))
        if matching and complete:
            print(f"Conversion already complete at {folder}. Existing files were kept.")
            print(f"Report: {folder / 'report.json'}")
            return
        if complete:
            raise ValueError(f"{folder} contains a completed conversion with different settings; "
                             "choose another --output path for a new run")
        if args.resume and not matching:
            raise ValueError(f"{folder} settings do not match this resume command")
        if not args.resume:
            raise ValueError(f"{folder} contains an incomplete conversion; "
                             "use --resume or choose another --output path")
    model, tokenizer, revision = load_dense(
        args.model, device=args.device, dtype=args.dtype)
    cfg = getattr(model.config, "text_config", model.config)
    if cfg.hidden_act != "silu" or cfg.intermediate_size % args.experts:
        raise ValueError("Unsupported FFN configuration")
    layers = list(range(cfg.num_hidden_layers))
    train = blocks_from_dataset(
        tokenizer, args.train_split, args.train_blocks, args.sequence_length,
        args.train_dataset, args.train_dataset_config, streaming=args.stream_train)
    validation = blocks_from_dataset(tokenizer, "validation", args.validation_blocks, args.sequence_length)
    test = blocks_from_dataset(tokenizer, "test", args.test_blocks, args.sequence_length)
    print("Evaluating dense baseline", flush=True)
    baseline = evaluate(model, test, args.device)
    dense_example = example_generation(model, tokenizer, args.device)
    print(json.dumps({"dense": baseline}), flush=True)
    if args.global_alignment and not args.online_teacher:
        print("Caching dense-teacher logits", flush=True)
        teacher_logits = []
        with torch.no_grad():
            for block in train:
                teacher_logits.append(model(input_ids=block[None].to(args.device),
                                            use_cache=False).logits.cpu().to(torch.float16))
        inputs = validation_inputs = None
    elif not args.global_alignment:
        print("Collecting calibration and validation activations", flush=True)
        inputs = collect_inputs(model, train, args.device)
        validation_inputs = collect_inputs(model, validation, args.device)
    else:
        print("Using exact all-expert outputs as the online dense teacher", flush=True)
        teacher_logits = None
        inputs = validation_inputs = None
    metadata = {"model_id": args.model, "revision": revision, "experts": args.experts,
                "top_k": args.top_k, "layers": layers, "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size, "settings": vars(args),
                "adapter_rank": args.adapter_rank,
                "status": "alignment_in_progress", "method": "layerwise OT + FFN MSE; frozen dense weights",
                "alignment_dataset": {
                    "id": args.train_dataset, "config": args.train_dataset_config,
                    "split": args.train_split, "streaming": args.stream_train},
                "evaluation_dataset": "Salesforce/wikitext / wikitext-2-raw-v1",
                "runtime": "custom PyTorch; original checkpoint plus this overlay required"}
    if args.shared_backbone:
        if args.top_k != 1 or args.adapter_rank < 1:
            raise ValueError("Shared-backbone conversion requires --top-k 1 and --adapter-rank >= 1")
        metadata["mode"] = "shared_backbone"
        metadata["method"] = "balanced token OT router + shared frozen dense FFN + expert adapters"
    if args.global_alignment:
        if args.shared_backbone or args.adapter_rank:
            raise ValueError("Global DOT alignment uses disjoint experts without residual adapters")
        metadata["method"] = "global DOT-MoE KL + CE + router z-loss + load balancing"
    (folder / "recipe.json").write_text(json.dumps(metadata, indent=2))
    layer_metrics = []
    if args.global_alignment:
        layer_metrics = global_dot_alignment(model, train, teacher_logits, args, args.device)
        (folder / "alignment_metrics.json").write_text(json.dumps(layer_metrics, indent=2))
    else:
      for index in layers:
        start = time.perf_counter()
        if args.shared_backbone:
            router, counts = balanced_token_router(inputs[index].to(args.device), args.experts)
            tensors = {
                "router": router,
                "adapter_up": torch.zeros(args.experts, args.adapter_rank, cfg.hidden_size),
                "adapter_down": torch.zeros(args.experts, cfg.hidden_size, args.adapter_rank),
            }
            metrics = {"initial_validation_relative_mse": 0.0,
                       "validation_relative_mse": 0.0,
                       "export_max_absolute_error": 0.0,
                       "calibration_tokens_per_expert": counts}
        else:
            tensors, metrics = train_layer(text_layers(model)[index].mlp, inputs[index],
                                          validation_inputs[index], args, args.device)
        save_file(tensors, str(folder / f"layer_{index:02d}.safetensors"))
        metrics.update(layer=index, seconds=time.perf_counter() - start)
        layer_metrics.append(metrics)
        print(json.dumps(metrics), flush=True)
        (folder / "alignment_metrics.json").write_text(json.dumps(layer_metrics, indent=2))
    if inputs is not None:
        del inputs, validation_inputs
    if args.global_alignment and teacher_logits is not None:
        del teacher_logits
    gc.collect()
    print("Installing learned experts and evaluating", flush=True)
    if not args.global_alignment:
        install_overlay(model, folder)
    converted = evaluate(model, test, args.device)
    converted_example = example_generation(model, tokenizer, args.device)
    report = {"dense": baseline, "moe": converted, "dense_generation": dense_example,
              "moe_generation": converted_example, "layers": layer_metrics,
              "ffn_active_fraction": 1.0 if args.shared_backbone else args.top_k / args.experts,
              "note": "Small text-only experiment; no broad capability or vision validation; timings include Python dispatch overhead"}
    if args.shared_backbone:
        report["note"] += "; one of eight adapters is active, while the full shared FFN backbone still runs"
    (folder / "report.json").write_text(json.dumps(report, indent=2))
    metadata["status"] = "converted_and_smoke_evaluated"
    (folder / "recipe.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    convert_parser = sub.add_parser("convert")
    convert_parser.add_argument("--model", default=MODEL_ID)
    convert_parser.add_argument("--output", default="qwen35_08b/moe")
    convert_parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    convert_parser.add_argument("--experts", type=int, default=8)
    convert_parser.add_argument("--top-k", type=int, default=7)
    convert_parser.add_argument("--steps", type=int, default=120)
    convert_parser.add_argument("--learning-rate", type=float, default=0.003)
    convert_parser.add_argument("--batch-tokens", type=int, default=64)
    convert_parser.add_argument("--batch-size", type=int, default=1,
                                help="Sequences per global-alignment optimizer step")
    convert_parser.add_argument("--train-blocks", type=int, default=16)
    convert_parser.add_argument("--validation-blocks", type=int, default=4)
    convert_parser.add_argument("--test-blocks", type=int, default=8)
    convert_parser.add_argument("--sequence-length", type=int, default=128)
    convert_parser.add_argument("--seed", type=int, default=42)
    convert_parser.add_argument("--adapter-rank", type=int, default=0,
                                help="Shared low-rank residual used to recover omitted expert output")
    convert_parser.add_argument("--shared-backbone", action="store_true",
                                help="Build exact 1-of-N routed adapters over the dense FFN")
    convert_parser.add_argument("--global-alignment", action="store_true",
                                help="Jointly train DOT assignments and routers with KL + CE")
    convert_parser.add_argument("--kl-weight", type=float, default=2.0)
    convert_parser.add_argument("--ce-weight", type=float, default=1.0)
    convert_parser.add_argument("--z-loss-weight", type=float, default=0.001)
    convert_parser.add_argument("--balance-weight", type=float, default=0.01)
    convert_parser.add_argument("--resume", action="store_true",
                                help="Resume an interrupted global alignment checkpoint")
    convert_parser.add_argument("--online-teacher", action="store_true",
                                help="Compute dense teacher logits online by activating all experts")
    convert_parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"),
                                default="float32")
    convert_parser.add_argument("--train-dataset", default="Salesforce/wikitext")
    convert_parser.add_argument("--train-dataset-config", default="wikitext-2-raw-v1")
    convert_parser.add_argument("--train-split", default="train")
    convert_parser.add_argument("--stream-train", action="store_true",
                                help="Stream alignment examples instead of downloading the dataset")
    recover_parser = sub.add_parser("recover")
    recover_parser.add_argument("--checkpoint", required=True)
    recover_parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    recover_parser.add_argument("--steps", type=int, default=200)
    recover_parser.add_argument("--learning-rate", type=float, default=0.0003)
    recover_parser.add_argument("--weight-decay", type=float, default=0.01)
    recover_parser.add_argument("--train-blocks", type=int, default=256)
    recover_parser.add_argument("--validation-blocks", type=int, default=16)
    recover_parser.add_argument("--test-blocks", type=int, default=16)
    recover_parser.add_argument("--sequence-length", type=int, default=64)
    recover_parser.add_argument("--eval-every", type=int, default=20)
    recover_parser.add_argument("--seed", type=int, default=42)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--checkpoint", default="qwen35_08b/moe")
    generate_parser.add_argument("--prompt", default="Explain optimal transport simply.")
    generate_parser.add_argument("--device", default="cpu")
    generate_parser.add_argument("--max-new-tokens", type=int, default=128)
    generate_parser.add_argument("--thinking", action="store_true",
                                 help="Enable the model's internal reasoning mode")
    generate_parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant. Give accurate, direct, complete answers.")
    generate_parser.add_argument("--top-k", type=int, default=None,
                                 help="Use the trained count or all experts")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", default="qwen35_08b/moe")
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.add_argument("--top-k", type=int, default=None)
    evaluate_parser.add_argument("--test-blocks", type=int, default=8)
    evaluate_parser.add_argument("--sequence-length", type=int, default=128)
    args = parser.parse_args()
    quiet_library_output()
    if args.command == "convert":
        try:
            convert(args)
        except ValueError as error:
            parser.error(str(error))
    elif args.command == "recover":
        try:
            recover_shared_backbone(args)
        except ValueError as error:
            parser.error(str(error))
    else:
        requested_checkpoint = Path(args.checkpoint)
        if args.top_k in (1, 2) and requested_checkpoint.name == "moe":
            exact_profile = requested_checkpoint.parent / f"moe_top{args.top_k}_exact"
            if exact_profile.is_dir():
                args.checkpoint = str(exact_profile)
        metadata = json.loads((Path(args.checkpoint) / "recipe.json").read_text())
        if metadata["status"] not in ("converted_and_smoke_evaluated", "recovered_and_evaluated"):
            raise ValueError("Conversion checkpoint is incomplete")
        requested_k = metadata["top_k"] if args.top_k is None else args.top_k
        if requested_k not in (metadata["top_k"], metadata["experts"]):
            parser.error(
                f"{args.checkpoint} was trained for --top-k {metadata['top_k']}. "
                f"Use --top-k {metadata['top_k']} or {metadata['experts']}; "
                "use a separately trained checkpoint for another value.")
        model, tokenizer, _ = load_dense(
            metadata["model_id"], metadata["revision"], args.device,
            metadata.get("settings", {}).get("dtype", "float32"))
        try:
            install_overlay(model, args.checkpoint, args.top_k)
        except ValueError as error:
            parser.error(str(error))
        if args.command == "evaluate":
            test = blocks_from_dataset(tokenizer, "test", args.test_blocks, args.sequence_length)
            result = evaluate(model, test, args.device)
            result["top_k"] = metadata["top_k"] if args.top_k is None else args.top_k
            result["experts"] = metadata["experts"]
            print(json.dumps(result, indent=2))
            return
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=args.thinking)
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
