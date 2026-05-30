"""Energy-calibrated gating for representation interventions.

Adapted from "Multi-Adapter Representation Interventions via Energy Calibration"
(MARI), https://arxiv.org/abs/2605.28722.

MARI observes that applying a *fixed* representation intervention uniformly to
every input degrades general capabilities on benign samples, because the
appropriate intervention direction and strength vary substantially across
inputs. Its remedy has two ingredients: a competitive multi-expert correction
and an energy-based gating module that uses the model's internal propagation
dynamics to decide which inputs an intervention should actually apply to.

This module delivers the energy-calibration ingredient as a small, reusable
gate that can be attached to any representation-intervention unit (e.g. ReFT).
The gate scores the incoming hidden state with a lightweight multi-expert head,
folds those scores into a Helmholtz free-energy value, and maps that energy
through a learnable affine + sigmoid into a per-sample, per-position scaling
factor in ``[0, 1]``. Low-energy (in-distribution / "applicable") states are
scaled toward ``1`` so the intervention fires at full strength; high-energy
states are scaled toward ``0`` so benign inputs pass through largely untouched.
"""

from typing import Optional

import torch
import torch.nn as nn


class EnergyCalibratedGate(nn.Module):
    """Energy-based gate that adaptively scales a representation intervention.

    Args:
        in_features: Hidden size of the states being gated.
        num_experts: Number of competitive scoring experts. The free energy is
            computed over this set of expert logits, so >1 lets the gate model
            non-linear "applicability" boundaries rather than a single
            threshold direction.
        temperature: Temperature ``T`` of the free-energy ``-T * logsumexp``.
            Higher values smooth the energy across experts.
        init_bias: Initial value of the learnable energy threshold. The gate is
            ``sigmoid(scale * (bias - energy))``; ``bias`` is where the gate
            crosses 0.5 at initialization.
        dtype: Optional dtype for the gate parameters.
    """

    def __init__(
        self,
        in_features: int,
        num_experts: int = 4,
        temperature: float = 1.0,
        init_bias: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.temperature = float(temperature)
        # Lightweight multi-expert scoring head; its logits define the energy.
        self.score_head = nn.Linear(in_features, num_experts, dtype=dtype)
        # Learnable affine calibration of the energy -> gate mapping.
        param_dtype = dtype or torch.float32
        self.gate_scale = nn.Parameter(torch.ones((), dtype=param_dtype))
        self.gate_bias = nn.Parameter(torch.full((), float(init_bias), dtype=param_dtype))

    def free_energy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Helmholtz free energy of the expert score distribution.

        Returns a tensor shaped like ``hidden_states`` without its last (hidden)
        dimension. Confident, in-distribution states yield a peaked expert
        distribution and therefore *low* (very negative) energy.
        """
        logits = self.score_head(hidden_states)
        return -self.temperature * torch.logsumexp(logits / self.temperature, dim=-1)

    def gate_values(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Per-position gate scalars in ``[0, 1]`` (no trailing hidden dim)."""
        energy = self.free_energy(hidden_states)
        scale = self.gate_scale.to(energy.dtype)
        bias = self.gate_bias.to(energy.dtype)
        # Low energy => applicable input => gate toward 1.
        return torch.sigmoid(scale * (bias - energy))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gate broadcastable against ``hidden_states`` (trailing dim of 1)."""
        return self.gate_values(hidden_states).unsqueeze(-1)
