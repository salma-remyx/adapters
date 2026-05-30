import torch
from transformers.testing_utils import require_torch

# Import from existing (non-new) modules to prove the integration is wired in.
from adapters import LoReftConfig
from adapters.methods.reft import ReftModule, ReftUnit


@require_torch
class TestEnergyGatedReft:
    in_dim = 16
    r_dim = 4

    def _unit(self, energy_gating):
        return ReftUnit(
            self.in_dim,
            self.r_dim,
            orthogonal=False,
            subtract_projection=True,
            non_linearity=None,
            dropout=0.0,
            energy_gating=energy_gating,
            gate_experts=4,
            gate_temperature=1.0,
        )

    def test_gate_attached_only_when_enabled(self):
        assert self._unit(False).gate is None
        gated = self._unit(True)
        assert gated.gate is not None

    def test_gate_values_in_unit_interval(self):
        unit = self._unit(True)
        unit.eval()
        x = torch.randn(2, 5, self.in_dim)
        gate = unit.gate.gate_values(x)
        # one scalar per (batch, position), all within [0, 1]
        assert gate.shape == (2, 5)
        assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)

    def test_gating_scales_the_intervention(self):
        # Same weights for gated and ungated unit so the only difference is the gate.
        ungated = self._unit(False)
        gated = self._unit(True)
        gated.load_state_dict(ungated.state_dict(), strict=False)
        gated.eval()
        ungated.eval()

        x = torch.randn(3, 4, self.in_dim)
        delta_ungated = ungated(x) - x
        delta_gated = gated(x) - x

        gate = gated.gate(x)  # (3, 4, 1)
        # Gated delta is exactly the ungated delta scaled by the per-position gate.
        assert torch.allclose(delta_gated, gate * delta_ungated, atol=1e-5)
        # And the gate genuinely attenuates: |gated delta| never exceeds ungated.
        assert torch.all(delta_gated.abs() <= delta_ungated.abs() + 1e-5)

    def test_reft_module_builds_gated_units_from_config(self):
        config = LoReftConfig(energy_gating=True, gate_experts=3, gate_temperature=2.0)
        module = ReftModule(self.in_dim, config)
        assert len(module.units) > 0
        for unit in module.units:
            assert isinstance(unit, ReftUnit)
            assert unit.gate is not None
            assert unit.gate.temperature == 2.0
            assert unit.gate.score_head.out_features == 3

    def test_default_config_leaves_units_ungated(self):
        config = LoReftConfig()
        module = ReftModule(self.in_dim, config)
        for unit in module.units:
            assert unit.gate is None
