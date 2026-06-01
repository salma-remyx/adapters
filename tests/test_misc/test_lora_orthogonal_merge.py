import unittest

import torch

import adapters
from adapters import LoRAConfig
from adapters.methods.lora import LoRALayer
from adapters.methods.lora_orthogonal_merge import (
    orthogonal_merge_lora,
    orthogonalize_deltas,
    residual_interference_energy,
)
from transformers import BertConfig, BertForSequenceClassification
from transformers.testing_utils import require_torch, torch_device


@require_torch
class LoRAOrthogonalMergeTest(unittest.TestCase):
    """Tests the SeqLoRA-inspired orthogonal-subspace merge strategy and its
    wiring into the library's existing ``average_adapter`` call site."""

    def build_model(self):
        model = BertForSequenceClassification(
            BertConfig(
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=37,
            )
        )
        adapters.init(model)
        return model

    def test_residual_interference_energy_zero_for_orthogonal(self):
        # Two updates with disjoint output rows are orthogonal -> zero interference.
        a = torch.zeros(4, 4)
        a[0, 0] = 1.0
        b = torch.zeros(4, 4)
        b[1, 1] = 1.0
        self.assertAlmostEqual(residual_interference_energy([a, b]).item(), 0.0, places=6)
        # Identical updates maximally interfere -> strictly positive.
        self.assertGreater(residual_interference_energy([a, a]).item(), 0.0)

    def test_orthogonalize_deltas_removes_interference(self):
        torch.manual_seed(0)
        deltas = [torch.randn(8, 6) for _ in range(3)]
        before = residual_interference_energy(deltas).item()
        ortho = orthogonalize_deltas(deltas)
        after = residual_interference_energy(ortho).item()
        # The first update is untouched; later ones are deflated into the orthogonal complement.
        self.assertTrue(torch.allclose(ortho[0], deltas[0]))
        self.assertGreater(before, 0.0)
        self.assertLess(after, before * 1e-6 + 1e-6)

    def test_orthogonal_merge_lora_factor_shapes_and_energy(self):
        torch.manual_seed(0)
        d, k, rank = 12, 10, 4
        deltas = [torch.randn(d, k) for _ in range(3)]
        weights = [0.5, 0.3, 0.2]
        state_dict, energy_before, energy_after = orthogonal_merge_lora(deltas, weights, rank=rank)
        self.assertEqual(tuple(state_dict["lora_A"].shape), (rank, k))
        self.assertEqual(tuple(state_dict["lora_B"].shape), (d, rank))
        # The orthogonal merge measurably reduces residual interference energy.
        self.assertGreater(energy_before.item(), energy_after.item())

    def test_average_adapter_accepts_orthogonal_strategy(self):
        # Exercises the wiring edit in adapters.methods.lora / model_mixin: the strategy
        # used to be rejected by _pre_average_adapter_checks and now merges through.
        model = self.build_model()
        model.to(torch_device)

        config = LoRAConfig(r=4)
        names = ["concept_a", "concept_b", "concept_c"]
        for name in names:
            model.add_adapter(name, config=config)

        model.average_adapter(
            "merged",
            names,
            weights=[0.5, 0.3, 0.2],
            combine_strategy="lora_orthogonal",
        )

        # Merged adapter is registered and produces usable LoRA factors of the right rank.
        self.assertIn("merged", model.adapters_config)
        merged_modules = model.get_adapter("merged")
        self.assertGreater(len(merged_modules), 0)

        checked = 0
        for module in model.modules():
            if isinstance(module, LoRALayer) and "merged" in module.loras:
                lora = module.loras["merged"]
                self.assertEqual(lora.lora_A.shape[0], 4)
                self.assertEqual(lora.lora_B.shape[-1], 4)
                checked += 1
        self.assertGreater(checked, 0)

    def test_orthogonal_merge_matches_linear_factor_shapes(self):
        # End-to-end parity: the orthogonal merge yields a "merged" adapter whose LoRA
        # factor shapes match those produced by the existing linear merge strategy.
        model = self.build_model()
        config = LoRAConfig(r=4, init_weights="bert")
        names = ["s0", "s1", "s2"]
        for name in names:
            model.add_adapter(name, config=config)

        model.average_adapter("linear_merge", names, combine_strategy="linear")
        model.average_adapter("orth_merge", names, combine_strategy="lora_orthogonal")

        linear_shapes, orth_shapes = {}, {}
        for module in model.modules():
            if isinstance(module, LoRALayer) and "linear_merge" in module.loras:
                lin = module.loras["linear_merge"]
                orth = module.loras["orth_merge"]
                linear_shapes[id(module)] = (tuple(lin.lora_A.shape), tuple(lin.lora_B.shape))
                orth_shapes[id(module)] = (tuple(orth.lora_A.shape), tuple(orth.lora_B.shape))

        self.assertGreater(len(orth_shapes), 0)
        self.assertEqual(linear_shapes, orth_shapes)


if __name__ == "__main__":
    unittest.main()
