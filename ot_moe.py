"""A small, architecture-independent DOT-MoE example for a gated FFN.

This converts one dense FFN; a transformer conversion replaces selected FFNs
and aligns their routers on hidden states from a calibration corpus.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sinkhorn(logits: torch.Tensor, temperature: float = 0.3, steps: int = 50) -> torch.Tensor:
    """Soft neuron-to-expert plan: rows sum to 1, columns to N / E."""
    n, experts = logits.shape
    if temperature <= 0 or steps < 1:
        raise ValueError("Temperature and Sinkhorn iteration count must be positive")
    if n % experts:
        raise ValueError("The intermediate width must be divisible by expert count")
    log_plan = logits.float() / temperature
    log_plan = log_plan - log_plan.max()
    target_column = torch.log(torch.tensor(n / experts, device=logits.device))
    for _ in range(steps):
        log_plan = log_plan - torch.logsumexp(log_plan, dim=1, keepdim=True)
        log_plan = log_plan - torch.logsumexp(log_plan, dim=0, keepdim=True) + target_column
    return log_plan.exp()


@torch.no_grad()
def balanced_round(plan: torch.Tensor) -> torch.Tensor:
    """Greedily turn the soft plan into an equal-capacity binary assignment."""
    n, experts = plan.shape
    capacity = n // experts
    if n % experts:
        raise ValueError("Expert capacity must be an integer")
    assignment = np.zeros((n, experts), dtype=np.float32)
    remaining = [capacity] * experts
    assigned = [False] * n
    assigned_count = 0
    order = np.argsort(-plan.detach().float().cpu().numpy().ravel(), kind="stable")
    for flat_index in order:
        neuron, expert = divmod(flat_index, experts)
        if not assigned[neuron] and remaining[expert] > 0:
            assignment[neuron, expert] = 1
            assigned[neuron] = True
            assigned_count += 1
            remaining[expert] -= 1
            if assigned_count == n:
                break
    return torch.from_numpy(assignment).to(device=plan.device, dtype=plan.dtype)


class GatedFFN(nn.Module):
    def __init__(self, hidden: int, intermediate: int, *, device=None, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AlignmentFFN(nn.Module):
    """Training-time masked dense FFN with learnable OT plan and router."""

    def __init__(self, dense: GatedFFN, experts: int, top_k: int, adapter_rank: int = 0):
        super().__init__()
        intermediate, hidden = dense.gate_proj.weight.shape
        if any(getattr(dense, name).bias is not None for name in ("gate_proj", "up_proj", "down_proj")):
            raise ValueError("This converter requires bias-free gated SiLU FFNs")
        if intermediate % experts or not 1 <= top_k <= experts:
            raise ValueError("Require equal-size experts and 1 <= top_k <= experts")
        self.dense = copy.deepcopy(dense)
        self.dense.requires_grad_(False)
        device = dense.gate_proj.weight.device
        self.affinity = nn.Parameter(torch.randn(intermediate, experts, device=device) * 0.01)
        self.router = nn.Linear(
            hidden, experts, bias=False, device=device, dtype=dense.gate_proj.weight.dtype)
        nn.init.normal_(self.router.weight, std=0.01)
        self.residual_up = None
        self.residual_down = None
        if adapter_rank:
            self.residual_up = nn.Linear(hidden, adapter_rank, bias=False, device=device)
            self.residual_down = nn.Linear(adapter_rank, hidden, bias=False, device=device)
            nn.init.kaiming_uniform_(self.residual_up.weight, a=5 ** 0.5)
            nn.init.zeros_(self.residual_down.weight)
        self.experts = experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor, temperature: float | None = None):
        model_call = temperature is None
        if temperature is None:
            temperature = getattr(self, "temperature", 0.3)
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        soft = sinkhorn(self.affinity, temperature)
        hard = balanced_round(soft)
        assignment = hard + (soft - soft.detach())  # straight-through estimator
        logits = F.linear(flat.float(), self.router.weight.float())
        probabilities = logits.softmax(dim=-1)
        indices = probabilities.topk(self.top_k, dim=-1).indices
        hard_route = torch.zeros_like(probabilities).scatter_(1, indices, 1.0)
        route = hard_route + (probabilities - probabilities.detach())
        active_neurons = route @ assignment.T
        activation = F.silu(self.dense.gate_proj(flat)) * self.dense.up_proj(flat)
        output = self.dense.down_proj(activation * active_neurons.to(activation.dtype))
        if self.residual_up is not None:
            residual = self.residual_down(F.silu(self.residual_up(flat.float())))
            output = output + residual.to(output.dtype)
        usage = hard_route.mean(dim=0) / self.top_k
        balance = self.experts * (usage * probabilities.mean(dim=0)).sum()
        z_loss = logits.logsumexp(dim=-1).square().mean()
        self.last_balance = balance
        self.last_z_loss = z_loss
        result = output.reshape(shape)
        return result if model_call else (result, balance, z_loss)

    @torch.no_grad()
    def export(self, temperature: float = 0.3):
        partition = balanced_round(sinkhorn(self.affinity, temperature))
        return SparseGatedMoE(self.dense, self.router, partition, self.top_k,
                              self.residual_up, self.residual_down)


class SparseGatedMoE(nn.Module):
    """Inference module computing only the selected expert FFNs."""

    def __init__(self, dense: GatedFFN, router: nn.Linear, partition: torch.Tensor, top_k: int,
                 residual_up: nn.Linear | None = None,
                 residual_down: nn.Linear | None = None):
        super().__init__()
        self.router = copy.deepcopy(router)
        self.top_k = top_k
        self.trained_top_k = top_k
        self.residual_up = copy.deepcopy(residual_up)
        self.residual_down = copy.deepcopy(residual_down)
        self.experts = nn.ModuleList()
        for expert in range(partition.shape[1]):
            neurons = partition[:, expert].nonzero(as_tuple=True)[0]
            hidden = dense.gate_proj.in_features
            sub = GatedFFN(hidden, len(neurons), device=dense.gate_proj.weight.device,
                           dtype=dense.gate_proj.weight.dtype)
            sub.gate_proj.weight.data.copy_(dense.gate_proj.weight.data[neurons])
            sub.up_proj.weight.data.copy_(dense.up_proj.weight.data[neurons])
            sub.down_proj.weight.data.copy_(dense.down_proj.weight.data[:, neurons])
            self.experts.append(sub)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        chosen = F.linear(flat.float(), self.router.weight.float()).topk(self.top_k, dim=-1).indices
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            rows = (chosen == expert_index).any(dim=-1).nonzero(as_tuple=True)[0]
            if rows.numel():
                output.index_add_(0, rows, expert(flat.index_select(0, rows)))
        if self.residual_up is not None and self.top_k < len(self.experts):
            denominator = len(self.experts) - self.trained_top_k
            residual_scale = (len(self.experts) - self.top_k) / max(denominator, 1)
            residual = self.residual_down(F.silu(self.residual_up(flat.float())))
            output = output + residual.to(output.dtype) * residual_scale
        return output.reshape(shape)


class SharedBackboneMoE(nn.Module):
    """Top-k experts made from one exact dense backbone plus routed adapters.

    The backbone is evaluated once.  Each token then selects expert-specific
    low-rank residual parameters.  Zero-initialized residuals make conversion
    exactly equivalent to the source FFN while leaving room for specialization.
    """

    def __init__(self, dense: GatedFFN, router_weight: torch.Tensor,
                 adapter_up: torch.Tensor, adapter_down: torch.Tensor, top_k: int = 1):
        super().__init__()
        experts, rank, hidden = adapter_up.shape
        if adapter_down.shape != (experts, hidden, rank):
            raise ValueError("Invalid expert adapter shapes")
        if router_weight.shape != (experts, hidden):
            raise ValueError("Invalid router shape")
        if not 1 <= top_k <= experts:
            raise ValueError("Invalid active expert count")
        self.dense = copy.deepcopy(dense)
        self.dense.requires_grad_(False)
        self.router = nn.Linear(hidden, experts, bias=False, device=dense.gate_proj.weight.device)
        self.router.weight.data.copy_(router_weight.to(self.router.weight))
        self.adapter_up = nn.Parameter(adapter_up.to(self.router.weight), requires_grad=False)
        self.adapter_down = nn.Parameter(adapter_down.to(self.router.weight), requires_grad=False)
        self.top_k = top_k
        self.expert_count = experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        output = self.dense(flat)
        chosen = self.router(flat.float()).topk(self.top_k, dim=-1).indices
        residual = torch.zeros_like(flat)
        for expert_index in range(self.expert_count):
            rows = (chosen == expert_index).any(dim=-1).nonzero(as_tuple=True)[0]
            if rows.numel():
                selected = flat.index_select(0, rows).float()
                hidden = F.silu(F.linear(selected, self.adapter_up[expert_index]))
                update = F.linear(hidden, self.adapter_down[expert_index])
                residual.index_add_(0, rows, update.to(residual.dtype))
        return (output + residual).reshape(shape)


def align_ffn(
    dense: GatedFFN,
    calibration: torch.Tensor,
    experts: int,
    top_k: int,
    steps: int = 300,
    learning_rate: float = 0.003,
) -> SparseGatedMoE:
    """Fit partition/router to dense FFN outputs on representative hidden states."""
    model = AlignmentFFN(dense, experts, top_k).to(calibration.device)
    optimizer = torch.optim.AdamW([model.affinity, *model.router.parameters()], lr=learning_rate)
    with torch.no_grad():
        target = model.dense(calibration)
    for step in range(steps):
        temperature = 1.0 - 0.9 * (step / max(steps - 1, 1))
        prediction, balance, z_loss = model(calibration, temperature)
        loss = F.mse_loss(prediction, target) + 0.01 * balance + 0.001 * z_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.export(0.1)


if __name__ == "__main__":
    torch.manual_seed(0)
    dense_ffn = GatedFFN(hidden=32, intermediate=64)
    hidden_states = torch.randn(256, 32)
    moe = align_ffn(dense_ffn, hidden_states, experts=4, top_k=2)
    with torch.no_grad():
        relative_error = (moe(hidden_states) - dense_ffn(hidden_states)).norm() / dense_ffn(hidden_states).norm()
    print(f"Relative FFN output error on calibration data: {relative_error.item():.3f}")
    print("Dense intermediate neurons per token: 64; MoE: 32")
