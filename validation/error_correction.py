# validation/error_correction.py
"""Error correction pressure test: quantify prior's ability to suppress false positives.

Compares Tempered HGT vs Pure HGT on injected false-positive edges (Hmgb1/Padi4).
Hypothesis: Tempered HGT's negative-anchor prior assigns LOWER scores to
known-false edges compared to Pure HGT (no prior).
"""
import torch
from typing import Dict, List, Tuple
from torch_geometric.data import HeteroData


class ErrorCorrectionTest:
    """Quantify error correction capability of the three-layer prior.

    Usage:
        test = ErrorCorrectionTest(injected_edges, model_names)
        scores = test.run(models, data, edge_index_dict, cos_decay_map)
    """

    def __init__(self, injected_edges: List[Tuple],
                 model_names: List[str]):
        self.injected_edges = injected_edges
        self.model_names = model_names

    def run(self, models: Dict[str, torch.nn.Module],
            data: HeteroData,
            edge_index_dict: Dict[Tuple, torch.Tensor],
            cos_decay_map: Dict[str, float]) -> Dict[str, dict]:
        """Score injected false-positive edges under each model.

        Returns:
            {"Model Name": {"fp_scores": [float], "mean_fp_score": float,
                            "median_fp_score": float, "max_fp_score": float}}
        """
        results = {}

        for name in self.model_names:
            model = models[name]
            model.eval()
            cos_decay = cos_decay_map.get(name, 0.0)

            x_dict = {}
            for nt in data.node_types:
                if hasattr(data[nt], 'x') and data[nt].x is not None:
                    x_dict[nt] = data[nt].x
                else:
                    x_dict[nt] = torch.randn(data[nt].num_nodes, 896)

            with torch.no_grad():
                z_dict = model(x_dict, edge_index_dict, cos_decay=cos_decay)

            fp_scores = []
            for et, src_idx, dst_idx in self.injected_edges:
                src_t, rel, dst_t = et
                edge = torch.tensor([[src_idx], [dst_idx]])
                score = model.decode(z_dict, edge, src_t, dst_t).item()
                fp_scores.append(score)

            fp_scores_t = torch.tensor(fp_scores)
            results[name] = {
                "fp_scores": fp_scores,
                "mean_fp_score": fp_scores_t.mean().item(),
                "median_fp_score": fp_scores_t.median().item(),
                "max_fp_score": fp_scores_t.max().item(),
            }

        return results
