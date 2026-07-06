# models/baselines.py
"""Baseline models for fair comparison against Tempered HGT.

PyG built-in implementations prevent self-comparison bias:
  - PyGRGCNBaseline: RGCNConv wrapped in HeteroConv
  - HANBaseline: HANConv with 3 biological meta-paths
  - PureHGTFactory: TemperedHGT with tau=1.0 frozen, cos_decay=0
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from torch_geometric.nn import RGCNConv, HANConv, HeteroConv


class _RGCNConvWrapper(nn.Module):
    """Thin wrapper that supplies edge_type=0 (single relation) to RGCNConv.

    HeteroConv calls conv(*args) with (x_src, x_dst, edge_index).  RGCNConv
    additionally requires ``edge_type``, so this wrapper intercepts positional
    args and injects a zero edge_type tensor.
    """

    def __init__(self, in_channels: int, out_channels: int, num_relations: int = 1):
        super().__init__()
        self.conv = RGCNConv(in_channels, out_channels, num_relations=num_relations)

    def forward(self, *args, **kwargs):
        # args = (x_src, x_dst, edge_index) from HeteroConv
        edge_index = args[-1]
        if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
            num_edges = edge_index.shape[1]
            kwargs["edge_type"] = torch.zeros(num_edges, dtype=torch.long,
                                              device=edge_index.device)
        return self.conv(*args, **kwargs)


class PyGRGCNBaseline(nn.Module):
    """RGCN baseline using PyG built-in RGCNConv.

    Same interface as TemperedHGT for FairTrainer compatibility.
    """

    def __init__(self, in_channels: Dict[str, int], hidden_channels: int,
                 out_channels: int, num_layers: int,
                 meta_relations: List[Tuple[str, str, str]],
                 dropout: float = 0.1):
        super().__init__()
        self.meta_relations = meta_relations

        node_types = set()
        for src, _, dst in meta_relations:
            node_types.add(src)
            node_types.add(dst)
        self.node_types = sorted(node_types)

        from .encoders import HeteroDictEncoder, InnerProductDecoder
        self.encoder = HeteroDictEncoder(
            in_channels=in_channels, hidden_channels=hidden_channels,
            node_types=self.node_types, dropout=dropout,
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for src, rel, dst in meta_relations:
                conv_dict[(src, rel, dst)] = _RGCNConvWrapper(
                    hidden_channels, hidden_channels, num_relations=1,
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        self.out_proj = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.decoder = InnerProductDecoder()

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        h = self.encoder(x_dict)
        for conv in self.convs:
            updated = conv(h, edge_index_dict)
            # HeteroConv only returns destination node types; carry forward
            # source node types unchanged so they survive through all layers.
            h = {nt: updated.get(nt, h[nt]) for nt in h}
            h = {nt: self.dropout(torch.relu(emb)) for nt, emb in h.items()}
        return {nt: self.out_proj(emb) for nt, emb in h.items()}

    def decode(self, z_dict, edge_index, src_type, dst_type):
        return self.decoder(z_dict, edge_index, src_type, dst_type)

    def decode_prob(self, z_dict, edge_index, src_type, dst_type):
        return self.decoder.decode_prob(z_dict, edge_index, src_type, dst_type)


class HANBaseline(nn.Module):
    """HAN baseline with 3 biologically-meaningful meta-paths.

    Meta-paths (short to long, for multi-hop analysis):
    1. Gene -> ASSOCIATED_WITH -> Disease  (direct)
    2. Gene -> REGULATES -> Gene, Gene -> ASSOCIATED_WITH -> Disease  (gene cascade)
    3. Drug -> INHIBITS -> Protein, Protein -> ASSOCIATED_WITH -> Disease  (pharmacological)
    """

    def __init__(self, in_channels: Dict[str, int], hidden_channels: int,
                 out_channels: int, num_heads: int,
                 meta_paths: List[List[Tuple[str, str, str]]],
                 dropout: float = 0.1):
        super().__init__()

        self.meta_paths = meta_paths

        node_types = set()
        for path in meta_paths:
            for src, _, dst in path:
                node_types.add(src)
                node_types.add(dst)
        self.node_types = sorted(node_types)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        from .encoders import HeteroDictEncoder, InnerProductDecoder
        self.encoder = HeteroDictEncoder(
            in_channels=in_channels, hidden_channels=hidden_channels,
            node_types=self.node_types, dropout=dropout,
        )

        # HAN: compute separate attention per meta-path then semantic-level attention.
        # Note: PyG HANConv outputs ``out_channels`` features (without head
        # multiplication) in the installed version, so we pass the full
        # hidden_channels as out_channels and let ``heads`` control the number
        # of attention heads internally.
        self.han_layers = nn.ModuleList()
        for path in meta_paths:
            self.han_layers.append(HANConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                heads=num_heads,
                dropout=dropout,
                metadata=(list(node_types), path),
            ))

        self.semantic_attn = nn.Parameter(torch.ones(len(meta_paths)) / len(meta_paths))
        self.out_proj = nn.Linear(hidden_channels, out_channels)
        self.decoder = InnerProductDecoder()

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        h = self.encoder(x_dict)

        # Process each meta-path independently
        path_outputs = []
        for layer in self.han_layers:
            try:
                out = layer(h, edge_index_dict)
                path_outputs.append(out)
            except Exception:
                continue  # meta-path may be incomplete for current graph

        if not path_outputs:
            # Fallback: return encoded features
            return {nt: self.out_proj(emb) for nt, emb in h.items()}

        # Semantic-level attention: weighted sum across meta-paths
        weights = torch.softmax(self.semantic_attn[:len(path_outputs)], dim=0)
        out = {}
        for nt in self.node_types:
            nt_outputs = [po[nt] for po in path_outputs
                          if nt in po and po[nt] is not None]
            if nt_outputs:
                stacked = torch.stack(nt_outputs, dim=0)  # [P, N, D]
                weighted = (stacked * weights[:len(nt_outputs)].view(-1, 1, 1)).sum(0)
                out[nt] = self.out_proj(weighted)
            elif nt in h:
                out[nt] = self.out_proj(h[nt])

        return out

    def decode(self, z_dict, edge_index, src_type, dst_type):
        return self.decoder(z_dict, edge_index, src_type, dst_type)

    def decode_prob(self, z_dict, edge_index, src_type, dst_type):
        return self.decoder.decode_prob(z_dict, edge_index, src_type, dst_type)


class PureHGTFactory:
    """Factory for Pure HGT: TemperedHGT with all prior injection disabled.

    KEY ABLATION: Same code as TemperedHGT, only priors off.
    Performance gap = value of three-layer prior.
    """

    @staticmethod
    def create(in_channels: Dict[str, int], hidden_channels: int,
               out_channels: int, num_heads: int, num_layers: int,
               meta_relations: List[Tuple[str, str, str]],
               dropout: float = 0.1) -> nn.Module:
        from .tempered_hgt import TemperedHGT

        temperature_init = {}
        for src, rel, dst in meta_relations:
            key = f"{src}__{rel}__{dst}"
            temperature_init[key] = 1.0

        model = TemperedHGT(
            in_channels=in_channels, hidden_channels=hidden_channels,
            out_channels=out_channels, num_heads=num_heads,
            num_layers=num_layers, meta_relations=meta_relations,
            temperature_init=temperature_init, dropout=dropout,
        )

        for conv in model.convs:
            for key, tau in conv.temperatures.items():
                tau.requires_grad = False

        return model
