"""Projection encoders and link prediction decoders for heterogeneous GNNs."""
import warnings
import torch
import torch.nn as nn
from typing import Dict, List


class HeteroDictEncoder(nn.Module):
    """Per-node-type linear projection with LayerNorm.

    Maps raw features (896d from Phase 1) to a unified hidden dimension.
    """

    def __init__(self, in_channels: Dict[str, int], hidden_channels: int,
                 node_types: List[str], dropout: float = 0.1):
        super().__init__()
        self.node_types = node_types
        self.projections = nn.ModuleDict()
        self.norms = nn.ModuleDict()

        for nt in node_types:
            in_ch = in_channels.get(nt, hidden_channels)
            self.projections[nt] = nn.Linear(in_ch, hidden_channels)
            self.norms[nt] = nn.LayerNorm(hidden_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for nt, x in x_dict.items():
            if nt in self.projections:
                h = self.projections[nt](x)
                h = self.norms[nt](h)
                h = self.dropout(h)
                out[nt] = h
            else:
                warnings.warn(f"Unknown node type '{nt}' in encoder - using random projection")
                # Find any existing projection to get output dim
                any_proj = next(iter(self.projections.values()))
                proj = nn.Linear(x.shape[-1], any_proj.out_features).to(x.device)
                out[nt] = proj(x)
        return out


class InnerProductDecoder(nn.Module):
    """Dot-product decoder for Link Prediction.

    logits = (z_src * z_dst).sum(dim=-1)
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_dict: Dict[str, torch.Tensor],
                edge_index: torch.Tensor, src_type: str, dst_type: str) -> torch.Tensor:
        z_src = z_dict[src_type][edge_index[0]]
        z_dst = z_dict[dst_type][edge_index[1]]
        return (z_src * z_dst).sum(dim=-1)

    def decode_prob(self, z_dict: Dict[str, torch.Tensor],
                    edge_index: torch.Tensor, src_type: str, dst_type: str) -> torch.Tensor:
        return torch.sigmoid(self(z_dict, edge_index, src_type, dst_type))

    def score_all_pairs(self, z_dict: Dict[str, torch.Tensor],
                         src_type: str, dst_type: str) -> torch.Tensor:
        return torch.matmul(z_dict[src_type], z_dict[dst_type].T)
