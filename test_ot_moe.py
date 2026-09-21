import unittest
import tempfile
from argparse import Namespace
from pathlib import Path

import torch
from transformers import (Qwen2Config, Qwen2ForCausalLM, Qwen3_5Config,
                          Qwen3_5ForConditionalGeneration)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

from ot_moe import AlignmentFFN, SharedBackboneMoE, balanced_round, sinkhorn
from dot_moe_converter import derive_topology
from qwen_moe import global_dot_alignment


class ConversionTests(unittest.TestCase):
    def test_paper_topologies(self):
        self.assertEqual(derive_topology(18944, 128, 0.25), (148, 37))
        self.assertEqual(derive_topology(14336, 128, 0.25), (112, 28))
        self.assertEqual(derive_topology(11008, 128, 0.25), (86, 22))

    def test_global_paper_alignment_exports_every_layer(self):
        torch.manual_seed(5)
        config = Qwen2Config(
            hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=64)
        model = Qwen2ForCausalLM(config).eval()
        blocks = torch.randint(0, 64, (4, 8))
        with tempfile.TemporaryDirectory() as folder:
            args = Namespace(
                experts=4, top_k=2, learning_rate=5e-4, steps=2,
                batch_size=2, output=folder, resume=False, online_teacher=True,
                kl_weight=2.0, ce_weight=1.0, z_loss_weight=1e-3,
                balance_weight=1e-2)
            metrics = global_dot_alignment(model, blocks, None, args, "cpu")
            self.assertEqual(len(metrics), 2)
            for index, metric in enumerate(metrics):
                self.assertTrue((Path(folder) / f"layer_{index:02d}.safetensors").is_file())
                self.assertEqual(metric["expert_neuron_counts"], [8.0] * 4)

    def test_balanced_transport_and_gradients(self):
        logits = torch.randn(32, 4, requires_grad=True)
        plan = sinkhorn(logits, 0.5, steps=100)
        torch.testing.assert_close(plan.sum(1), torch.ones(32), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(plan.sum(0), torch.full((4,), 8.0))
        hard = balanced_round(plan)
        torch.testing.assert_close(hard.sum(1), torch.ones(32))
        torch.testing.assert_close(hard.sum(0), torch.full((4,), 8.0))
        (plan * torch.randn_like(plan)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_qwen_export_and_frozen_weights(self):
        torch.manual_seed(3)
        config = Qwen3_5TextConfig(hidden_size=16, intermediate_size=32)
        dense = Qwen3_5MLP(config, intermediate_size=32).eval()
        alignment = AlignmentFFN(dense, experts=4, top_k=2)
        x = torch.randn(2, 5, 16)
        output = alignment(x, 0.1)[0]
        output.square().mean().backward()
        self.assertGreater(alignment.affinity.grad.abs().sum().item(), 0)
        self.assertGreater(alignment.router.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in alignment.dense.parameters()))
        sparse = alignment.export(0.1)
        torch.testing.assert_close(sparse(x), output, atol=1e-6, rtol=1e-5)
        sparse.top_k = 4
        torch.testing.assert_close(sparse(x), dense(x), atol=1e-6, rtol=1e-5)

    def test_bfloat16_export_preserves_dtype(self):
        dense = Qwen3_5MLP(Qwen3_5TextConfig(hidden_size=16, intermediate_size=32), intermediate_size=32).bfloat16()
        alignment = AlignmentFFN(dense, 4, 4)
        self.assertEqual(alignment.router.weight.dtype, torch.bfloat16)
        sparse = alignment.export()
        self.assertEqual(sparse.experts[0].gate_proj.weight.dtype, torch.bfloat16)
        x = torch.randn(4, 16).bfloat16()
        torch.testing.assert_close(sparse(x), dense(x), atol=0.005, rtol=0.02)

    def test_residual_adapter_export_and_dense_fallback(self):
        dense = Qwen3_5MLP(Qwen3_5TextConfig(hidden_size=16, intermediate_size=32), 32).eval()
        alignment = AlignmentFFN(dense, experts=4, top_k=2, adapter_rank=4)
        torch.nn.init.normal_(alignment.residual_down.weight, std=0.02)
        x = torch.randn(7, 16)
        expected = alignment(x, 0.1)[0]
        sparse = alignment.export(0.1)
        torch.testing.assert_close(sparse(x), expected, atol=1e-6, rtol=1e-5)
        sparse.top_k = 4
        torch.testing.assert_close(sparse(x), dense(x), atol=1e-6, rtol=1e-5)

    def test_shared_experts_start_dense_and_can_learn(self):
        torch.manual_seed(4)
        dense = Qwen3_5MLP(
            Qwen3_5TextConfig(hidden_size=16, intermediate_size=32), 32).eval()
        router = torch.randn(4, 16)
        adapter_up = torch.randn(4, 3, 16) * 0.02
        adapter_down = torch.zeros(4, 16, 3)
        moe = SharedBackboneMoE(dense, router, adapter_up, adapter_down, top_k=1)
        moe.adapter_up.requires_grad_(True)
        moe.adapter_down.requires_grad_(True)
        x = torch.randn(9, 16)
        torch.testing.assert_close(moe(x), dense(x), atol=1e-6, rtol=1e-5)
        moe(x).square().mean().backward()
        self.assertGreater(moe.adapter_down.grad.abs().sum().item(), 0)

    def test_full_qwen_logits_after_all_expert_export(self):
        config = Qwen3_5Config(
            text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                linear_key_head_dim=16, linear_value_head_dim=16,
                linear_num_key_heads=2, linear_num_value_heads=2, vocab_size=256,
                layer_types=["linear_attention"] * 3 + ["full_attention"]),
            vision_config=dict(hidden_size=32, intermediate_size=64, depth=1,
                num_heads=4, out_hidden_size=64, patch_size=2, spatial_merge_size=2))
        model = Qwen3_5ForConditionalGeneration(config).eval()
        ids = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            expected = model(input_ids=ids, use_cache=False).logits
            for layer in model.model.language_model.layers:
                layer.mlp = AlignmentFFN(layer.mlp, 4, 4).export()
            actual = model(input_ids=ids, use_cache=False).logits
            torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-4)
            generated = model.generate(input_ids=ids, max_new_tokens=2,
                do_sample=False, pad_token_id=0, eos_token_id=None)
            self.assertEqual(generated.shape, (1, 18))


if __name__ == "__main__":
    unittest.main()
