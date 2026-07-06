# data/ablation_injection.py
"""Adversarial false-positive edge injection for ablation experiments.

Injects synthetic false-positive edges (Hmgb1->VTE, Padi4->DVT) into training
data to test whether the Tempered HGT prior can suppress known-false associations.
"""
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import HeteroData


class FalsePositiveInjector:
    """Inject adversarial false-positive edges into training data.

    Tests the "error correction" capability of the three-layer prior.
    """

    def __init__(self, config_path: str = "config/ablation_config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.injection_cfg = self.config.get("false_positive_injection", {})
        self.pairs = self.injection_cfg.get("pairs", [])

    def resolve_node_indices(self, data: HeteroData,
                              node_name_to_idx: Dict[str, Dict[str, int]],
                              ) -> List[Tuple[Tuple, int, int]]:
        """Resolve named false-positive pairs to (edge_type, src_idx, dst_idx)."""
        resolved = []
        for pair in self.pairs:
            src_t = pair["source_type"]
            dst_t = pair["target_type"]
            rel = pair["relation"]
            src_name = pair["source"]
            dst_name = pair["target"]

            et = (src_t, rel, dst_t)
            src_map = node_name_to_idx.get(src_t, {})
            dst_map = node_name_to_idx.get(dst_t, {})

            src_idx = src_map.get(src_name)
            dst_idx = dst_map.get(dst_name)

            if src_idx is None:
                for name, idx in src_map.items():
                    if src_name.lower() in name.lower():
                        src_idx = idx
                        break

            if dst_idx is None:
                for name, idx in dst_map.items():
                    if dst_name.lower() in name.lower():
                        dst_idx = idx
                        break

            if src_idx is not None and dst_idx is not None:
                resolved.append((et, src_idx, dst_idx))
            else:
                print(f"  Warning: Could not resolve {src_name}->{dst_name}")

        return resolved

    def inject(self, data: HeteroData,
               node_name_to_idx: Dict[str, Dict[str, int]],
               train_ei: Dict[Tuple, torch.Tensor],
               neg_ei: Optional[Dict[Tuple, torch.Tensor]] = None,
               ) -> Tuple[Dict, Dict, List[Tuple]]:
        """Inject false-positive edges into training data.

        Returns:
            (updated_train_ei, updated_neg_ei, injected_edges)
        """
        resolved = self.resolve_node_indices(data, node_name_to_idx)

        if not resolved:
            print("  No false-positive pairs resolved")
            return train_ei, neg_ei or {}, []

        updated_train = {k: v.clone() for k, v in train_ei.items()}
        updated_neg = {k: v.clone() for k, v in (neg_ei or {}).items()}
        injected_edges = []

        for et, src_idx, dst_idx in resolved:
            new_edge = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)

            if et in updated_train:
                updated_train[et] = torch.cat([updated_train[et], new_edge], dim=1)
            else:
                updated_train[et] = new_edge

            if et in updated_neg:
                neg_ei_typed = updated_neg[et]
                mask = ~((neg_ei_typed[0] == src_idx) & (neg_ei_typed[1] == dst_idx))
                updated_neg[et] = neg_ei_typed[:, mask]

            injected_edges.append((et, src_idx, dst_idx))

        return updated_train, updated_neg, injected_edges
