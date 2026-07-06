# tests/test_edge_bias.py
import pytest
import torch
from training.edge_bias import EdgeBiasInitializer, CosineAnnealingDecay


class TestEdgeBiasInitializer:
    @pytest.fixture
    def hard_priors_config(self):
        return {
            "verified_mechanisms": [
                {
                    "source_type": "Gene", "target_type": "Gene",
                    "relation": "REGULATES",
                    "pairs": [["FUT8", "Lgals3"], ["Lgals3", "CD44"]],
                    "initial_weight_multiplier": 3.0,
                    "decay": "cosine_annealing",
                }
            ],
            "pair_multiplier_override": {
                "FUT8_Lgals3": 5.0,
                "Lgals3_CD44": 4.0,
            },
        }

    @pytest.fixture
    def node_name_to_idx(self):
        return {
            "Gene": {"FUT8": 0, "Lgals3": 1, "CD44": 2, "Unknown": 3},
            "Disease": {"VTE": 0},
        }

    def test_build_bias_tensor_from_config(self, hard_priors_config, node_name_to_idx):
        initializer = EdgeBiasInitializer(hard_priors_config)
        bias_dict = initializer.build(node_name_to_idx)

        key = ("Gene", "REGULATES", "Gene")
        assert key in bias_dict

    def test_override_multiplier_takes_precedence(self, hard_priors_config, node_name_to_idx):
        initializer = EdgeBiasInitializer(hard_priors_config)
        initializer.build(node_name_to_idx)
        assert initializer._pair_overrides["FUT8_Lgals3"] == 5.0
        assert initializer._pair_overrides["Lgals3_CD44"] == 4.0

    def test_build_handles_empty_pairs(self, node_name_to_idx):
        config = {"verified_mechanisms": [], "pair_multiplier_override": {}}
        initializer = EdgeBiasInitializer(config)
        bias_dict = initializer.build(node_name_to_idx)
        assert bias_dict == {}

    def test_build_skips_unresolved_pairs(self, hard_priors_config):
        name_map = {"Gene": {"FUT8": 0}}  # Lgals3, CD44 not in map
        initializer = EdgeBiasInitializer(hard_priors_config)
        bias_dict = initializer.build(name_map)
        assert isinstance(bias_dict, dict)


class TestCosineAnnealingDecay:
    def test_decay_starts_at_one(self):
        schedule = CosineAnnealingDecay(total_steps=100)
        assert abs(schedule(0) - 1.0) < 1e-6

    def test_decay_ends_at_zero(self):
        schedule = CosineAnnealingDecay(total_steps=100)
        assert abs(schedule(100) - 0.0) < 1e-6

    def test_decay_monotonic_decrease(self):
        schedule = CosineAnnealingDecay(total_steps=50)
        values = [schedule(t) for t in range(0, 51, 10)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1], f"Not monotonic at step {i*10}"

    def test_decay_midpoint_is_half(self):
        schedule = CosineAnnealingDecay(total_steps=100)
        mid = schedule(50)
        assert abs(mid - 0.5) < 0.01

    def test_decay_handles_step_zero(self):
        schedule = CosineAnnealingDecay(total_steps=0)
        assert abs(schedule(0) - 1.0) < 1e-6

    def test_decay_formula_matches_cosine(self):
        import math
        schedule = CosineAnnealingDecay(total_steps=100)
        t = 25
        expected = 0.5 * (1 + math.cos(math.pi * t / 100))
        assert abs(schedule(t) - expected) < 1e-6
