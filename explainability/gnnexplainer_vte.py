"""Heterogeneous GNNExplainer wrapper for TemperedHGT link prediction.

CRITICAL: Model parameters FROZEN during explanation. Only edge/node masks optimized.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer


class VTEExplainer:
    def __init__(self, model: nn.Module, num_epochs: int = 200, lr: float = 0.01):
        self.model = model
        self.num_epochs = num_epochs
        self.lr = lr

    def _freeze_model(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _unfreeze_model(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def explain_edge(self, data: HeteroData, edge_type: Tuple, edge_idx: int,
                     x_dict: Dict[str, torch.Tensor]) -> Dict:
        self._freeze_model()
        src_t, rel, dst_t = edge_type

        def model_fn(x_dict, edge_index_dict):
            z = self.model(x_dict, edge_index_dict, cos_decay=1.0)
            ei = edge_index_dict[edge_type]
            return self.model.decode(z, ei[:, edge_idx:edge_idx+1], src_t, dst_t)

        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

        try:
            explainer = Explainer(
                model=model_fn,
                algorithm=GNNExplainer(epochs=self.num_epochs, lr=self.lr),
                explanation_type="model",
                model_config=dict(mode="regression", task_level="graph", return_type="raw"),
            )
            explanation = explainer(x_dict, edge_index_dict)
            result = {"edge_mask": explanation.get("edge_mask", {}), "edge_idx": edge_idx}
        except Exception:
            result = {"edge_mask": {}, "edge_idx": edge_idx}

        self._unfreeze_model()
        return result

    def explain_top_k(self, data: HeteroData, edge_type: Tuple, k: int = 10,
                      x_dict: Optional[Dict] = None) -> List[Dict]:
        if x_dict is None:
            x_dict = {}
            for nt in data.node_types:
                if hasattr(data[nt], 'x') and data[nt].x is not None:
                    x_dict[nt] = data[nt].x
                else:
                    x_dict[nt] = torch.randn(data[nt].num_nodes, 896)

        self.model.eval()
        with torch.no_grad():
            ei_dict = {et: data[et].edge_index for et in data.edge_types}
            z = self.model(x_dict, ei_dict, cos_decay=1.0)

        src_t, rel, dst_t = edge_type
        ei = data[edge_type].edge_index
        all_scores = self.model.decode(z, ei, src_t, dst_t)
        top_k = torch.topk(all_scores, min(k, len(all_scores)))
        return [{"edge_idx": idx.item(), "prediction_score": score.item()}
                for idx, score in zip(top_k.indices, top_k.values)]
