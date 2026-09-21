# DOT-MoE Converter

An independent PyTorch reproduction of
**[DOT-MoE: Differentiable Optimal Transport for MoEfication](https://arxiv.org/abs/2606.01666)**
(Bamba et al., ICML 2026), packaged as a dense-to-MoE converter for gated-SiLU
LLaMA/Qwen-style checkpoints.

This is an independent implementation, not the authors' official code.

## What is reproduced

For every dense FFN, the converter:

1. creates a learnable neuron-to-expert affinity matrix;
2. computes an equal-capacity soft assignment with 50 log-domain Sinkhorn
   iterations;
3. greedily rounds it to a disjoint, exactly balanced partition;
4. uses straight-through estimators for that partition and top-k token routing;
5. freezes the source weights and jointly trains only assignments and routers;
6. optimizes the paper's KL + language-model CE + router z-loss + load-balance
   objective; and
7. slices the original gate/up/down projections into standard sparse experts.

The saved checkpoint is a compact overlay: expert membership plus router
weights. It never modifies or redistributes the source model weights.

## Quick verification

```bash
git clone https://github.com/meloburr/dense-to-moe-converter.git
cd dense-to-moe-converter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest test_ot_moe -v
```

The tests cover balanced transport marginals, gradient flow through both STEs,
frozen dense weights, dtype preservation, masked-to-materialized export
equivalence, and full-model logits when every expert is active.

## Convert a model

The smoke profile validates the complete conversion pipeline with a small data
and step budget. It is not expected to reproduce the paper's scores:

```bash
./convert-dot-moe \
  --model Qwen/Qwen2.5-0.5B \
  --output outputs/qwen2.5-0.5b-dot-moe \
  --profile smoke \
  --device cpu \
  --dtype float32
```

Run the reported Qwen2.5-7B topology and alignment recipe with:

```bash
./convert-dot-moe \
  --model Qwen/Qwen2.5-7B \
  --output outputs/qwen2.5-7b-dot-moe \
  --profile paper \
  --device cuda \
  --dtype bfloat16
```

The paper profile derives 148 experts of 128 neurons and activates 37 per token
for Qwen2.5-7B. It uses 3,500 optimizer steps, sequence length 2,048, batch size
64, the reported loss weights, temperature annealing from 1.0 to 0.1, and the
streamed `allenai/dolmino-mix-1124` dataset. The paper reports this alignment as
taking under three hours for LLaMA-3-8B on 8 H100 GPUs; this reference runtime
is single-process and correctness-oriented, so comparable throughput requires
distributed training and fused expert kernels.

Use `--resume` after an interrupted run. A state checkpoint is written every 25
steps. `--offline-teacher` caches dense logits, which is faster but can consume
very large host memory; the default computes an exact online teacher by
temporarily activating all experts.

## Outputs

```text
outputs/qwen2.5-7b-dot-moe/
├── recipe.json
├── report.json
├── alignment_metrics.json
├── layer_00.safetensors
├── ...
└── layer_27.safetensors
```

Load, evaluate, or generate from an overlay with:

```bash
python qwen_moe.py evaluate --checkpoint outputs/qwen2.5-7b-dot-moe --device cuda
python qwen_moe.py generate \
  --checkpoint outputs/qwen2.5-7b-dot-moe \
  --device cuda \
  --prompt "Explain balanced optimal transport."
```

The custom Python dispatcher is a correctness implementation. The materialized
experts have the standard `[expert, intermediate, hidden]` structure needed for
a fused-MoE backend, but direct vLLM/llama.cpp/Ollama serialization is not yet
implemented.

## Paper-to-code map

| Paper component | Implementation |
|---|---|
| Eq. 7 / Algorithm 1, log-domain Sinkhorn | `ot_moe.py::sinkhorn` |
| Eq. 11, assignment STE | `AlignmentFFN.forward` |
| Eq. 12, top-k routing STE | `AlignmentFFN.forward` |
| Eq. 13, router z-loss | `AlignmentFFN.forward` |
| Eq. 14, load balancing | `AlignmentFFN.forward` |
| Eq. 15, total objective | `qwen_moe.py::global_dot_alignment` |
| Eq. 16, masked sparse computation | `AlignmentFFN.forward` |
| Expert materialization | `SparseGatedMoE` |
| Paper/scaled profiles | `dot_moe_converter.py` |

## Scope and limitations

- Supported FFNs must expose bias-free `gate_proj`, `up_proj`, and `down_proj`
  with SiLU activation and an intermediate width divisible by expert size.
- The published 1.2B-token continued fine-tuning stage is distinct from the
  3,500-step alignment and is not silently approximated here.
- Benchmark reproduction requires the paper's hardware budget and
  `lm-evaluation-harness` settings; a smoke run establishes implementation
  correctness, not the reported accuracy numbers.
- Model and dataset licenses remain their owners' licenses. This repository's
  source code is MIT licensed.

## Legacy experiment

`convert-better-moe` and `better_moe_converter.py` retain the earlier
quality-gated shared-backbone adapter experiment. That path runs the full dense
FFN and is intentionally separate from the paper-faithful, FLOP-reducing
`convert-dot-moe` command.
