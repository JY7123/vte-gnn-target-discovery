# training/edge_bias.py
"""Edge bias initialization from config and cosine annealing decay schedule."""
import math
import torch
import yaml
from typing import Dict, List, Optional, Tuple


class CosineAnnealingDecay:
    """Cosine annealing: f(t) = 0.5 * (1 + cos(pi * t / T)).

    Decays from 1.0 (full bias) at t=0 to 0.0 (no bias) at t=T.
    """

    def __init__(self, total_steps: int):
        self.total_steps = max(total_steps, 1)

    def __call__(self, step: int) -> float:
        if step >= self.total_steps:
            return 0.0
        if step <= 0:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * step / self.total_steps))


class EdgeBiasInitializer:
    """Build per-edge bias tensors from anchor_config.yaml hard_priors."""

    PAIR_SEP = "_"

    def __init__(self, hard_priors_config: dict):
        self.config = hard_priors_config
        self._pair_overrides = hard_priors_config.get("pair_multiplier_override", {})

    def _get_multiplier(self, src_name: str, dst_name: str, default: float) -> float:
        key = f"{src_name}{self.PAIR_SEP}{dst_name}"
        return self._pair_overrides.get(key, default)

    def build(self, node_name_to_idx: Dict[str, Dict[str, int]],
              edge_types: Optional[List[Tuple]] = None
              ) -> Dict[Tuple, torch.Tensor]:
        """Build edge bias dict mapping (src, rel, dst) -> list of (src_idx, dst_idx, multiplier)."""
        bias_dict = {}

        for mechanism in self.config.get("verified_mechanisms", []):
            src_t = mechanism["source_type"]
            dst_t = mechanism["target_type"]
            rel = mechanism["relation"]
            default_mult = mechanism.get("initial_weight_multiplier", 3.0)

            et = (src_t, rel, dst_t)
            if edge_types is not None and et not in edge_types:
                continue

            src_map = node_name_to_idx.get(src_t, {})
            dst_map = node_name_to_idx.get(dst_t, {})

            for src_name, dst_name in mechanism.get("pairs", []):
                src_idx = src_map.get(src_name)
                dst_idx = dst_map.get(dst_name)
                if src_idx is None or dst_idx is None:
                    continue

                multiplier = self._get_multiplier(src_name, dst_name, default_mult)

                if et not in bias_dict:
                    bias_dict[et] = []
                bias_dict[et].append((src_idx, dst_idx, multiplier))

        return bias_dict

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EdgeBiasInitializer":
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config.get("hard_priors", {}))
