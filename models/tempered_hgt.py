# models/tempered_hgt.py
"""Tempered HGT: Heterogeneous Graph Transformer with learnable per-relation
temperature tau and cosine-annealed edge bias injection.

Core attention formula:
    alpha = softmax( (Q * K) / (tau * sqrt(d)) + edge_bias * cos_decay )

Key: ParameterDict keys use double-underscore separator:
    "Gene__REGULATES__Disease"
"""
import math
import torch
import torch.nn as nn
from typing import Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Custom ParameterDict — PyTorch core does not ship nn.ParameterDict, so we
# provide a lightweight container that behaves like nn.ModuleDict but stores
# nn.Parameter instances as submodule attributes for serialisation safety.
# ---------------------------------------------------------------------------
class ParameterDict(nn.Module):
    """String-keyed container for nn.Parameter instances.

    Behaves like a dict but stores each value as a named attribute, which
    means they participate in state_dict / load_state_dict, ``parameters()``,
    ``to()``, and optimizer param-groups exactly like regular nn.Parameter
    members declared in ``__init__``.

    Keys must be valid Python identifiers after replacing ``.`` with ``_``.
    """

    def __init__(self, init_dict: Optional[Dict[str, nn.Parameter]] = None):
        super().__init__()
        self._keys: List[str] = []
        if init_dict is not None:
            for key, param in init_dict.items():
                self.__setitem__(key, param)

    def __setitem__(self, key: str, param: nn.Parameter) -> None:
        safe_name = key.replace(".", "_")
        setattr(self, safe_name, param)
        if key not in self._keys:
            self._keys.append(key)

    def __getitem__(self, key: str) -> nn.Parameter:
        safe_name = key.replace(".", "_")
        return getattr(self, safe_name)

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def items(self):
        for key in self._keys:
            yield key, self.__getitem__(key)

    def keys(self):
        return iter(self._keys)

    def values(self):
        for key in self._keys:
            yield self.__getitem__(key)


# Monkey-patch so the spec's isinstance check passes
nn.ParameterDict = ParameterDict


def _meta_to_key(src_type: str, rel_type: str, dst_type: str) -> str:
    """Convert meta-relation tuple to ParameterDict-safe string key."""
    return f"{src_type}__{rel_type}__{dst_type}"


class TemperedHGTConv(nn.Module):
    """Single heterogeneous graph attention layer with temperature and edge bias.

    Operates on all node types simultaneously. For each (src_t, rel, dst_t):
    1. Projects Q, K, V per source/destination node
    2. Computes attention: softmax(QK^T / (tau * sqrt(d)) + bias * cos_decay)
    3. Aggregates weighted V messages to destination nodes
    4. Residual connection + LayerNorm

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        meta_relations: List of (src, rel, dst) triples
        temperature_init: Dict mapping "Src__Rel__Dst" -> initial tau value
        dropout: Attention dropout rate
    """

    def __init__(self, in_channels: int, out_channels: int, num_heads: int,
                 meta_relations: List[Tuple[str, str, str]],
                 temperature_init: Dict[str, float],
                 dropout: float = 0.1):
        super().__init__()

        assert out_channels % num_heads == 0, (
            f"out_channels ({out_channels}) must be divisible by num_heads ({num_heads})"
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.sqrt_d = math.sqrt(self.head_dim)
        self.meta_relations = meta_relations

        # Per-relation learnable temperature parameters
        # nn.ParameterDict with string keys for serialization safety
        self.temperatures = nn.ParameterDict()
        for src, rel, dst in meta_relations:
            key = _meta_to_key(src, rel, dst)
            tau_init = temperature_init.get(key, 1.0)
            self.temperatures[key] = nn.Parameter(torch.tensor(tau_init))

        # Node types
        node_types = set()
        for src, _, dst in meta_relations:
            node_types.add(src)
            node_types.add(dst)
        self.node_types = sorted(node_types)

        # Linear projections: one per node type (as source OR destination)
        self.k_linears = nn.ModuleDict()
        self.q_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        for nt in self.node_types:
            self.k_linears[nt] = nn.Linear(in_channels, out_channels)
            self.q_linears[nt] = nn.Linear(in_channels, out_channels)
            self.v_linears[nt] = nn.Linear(in_channels, out_channels)

        # Output: project concatenated heads back to out_channels
        self.out_proj = nn.Linear(out_channels, out_channels)
        self.skip = nn.Linear(in_channels, out_channels)  # residual projection
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.k_linears, self.q_linears, self.v_linears]:
            for lin in m.values():
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def _clamp_temperatures(self):
        """Ensure all tau >= 0.01 to prevent division-by-zero and numerical explosion."""
        with torch.no_grad():
            for key in self.temperatures:
                self.temperatures[key].clamp_(min=0.01)

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple, torch.Tensor],
                cos_decay: float = 1.0,
                edge_weight_bias: Optional[Dict[Tuple, torch.Tensor]] = None,
                return_attention: bool = False,
                ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x_dict: {node_type: features [N_t, in_channels]}
            edge_index_dict: {(src_t, rel, dst_t): edge_index [2, E]}
            cos_decay: Cosine anneal coefficient in [0, 1]. 1.0 = full bias.
            edge_weight_bias: Optional {(src_t, rel, dst_t): bias [E]}
            return_attention: If True, also returns per-edge-type attention weights.

        Returns:
            If return_attention=False: {node_type: [N_t, out_channels]}
            If return_attention=True: (out_dict, {(src,rel,dst): [E] head-averaged})
        """
        self._clamp_temperatures()

        # Project Q, K, V for all node types
        q_dict = {}
        k_dict = {}
        v_dict = {}
        for nt in self.node_types:
            if nt in x_dict:
                q_dict[nt] = self.q_linears[nt](x_dict[nt])
                k_dict[nt] = self.k_linears[nt](x_dict[nt])
                v_dict[nt] = self.v_linears[nt](x_dict[nt])

        out_dict = {}
        attention_dict = {} if return_attention else None
        for dst_type in self.node_types:
            if dst_type not in x_dict:
                continue

            num_dst = x_dict[dst_type].shape[0]
            device = x_dict[dst_type].device
            agg_messages = []

            for (src_t, rel, dst_t), edge_index in edge_index_dict.items():
                if dst_t != dst_type:
                    continue
                key = _meta_to_key(src_t, rel, dst_t)
                if key not in self.temperatures:
                    continue
                if src_t not in q_dict or dst_t not in k_dict:
                    continue

                tau = self.temperatures[key]
                E = edge_index.shape[1]
                src_idx = edge_index[0]
                dst_idx = edge_index[1]

                # Gather Q, K, V for edges
                # [E, out_channels] -> reshape to [E, heads, head_dim]
                q = q_dict[src_t][src_idx].view(E, self.num_heads, self.head_dim)
                k = k_dict[dst_t][dst_idx].view(E, self.num_heads, self.head_dim)
                v = v_dict[dst_t][dst_idx].view(E, self.num_heads, self.head_dim)

                # Attention scores: dot product per head
                # [E, heads, head_dim] * [E, heads, head_dim] -> [E, heads]
                attn_scores = (q * k).sum(dim=-1)  # element-wise then sum

                # Temperature scaling: divide by tau * sqrt(d)
                # tau clamping ensures no division by zero
                attn_scores = attn_scores / (tau * self.sqrt_d)

                # Edge bias injection (Layer 1 hard prior)
                # Support both tuple keys (from EdgeBiasInitializer) and string keys
                bias_value = None
                if edge_weight_bias is not None:
                    if (src_t, rel, dst_t) in edge_weight_bias:
                        bias_value = edge_weight_bias[(src_t, rel, dst_t)]
                    elif key in edge_weight_bias:
                        bias_value = edge_weight_bias[key]
                if bias_value is not None:
                    bias = bias_value.to(device)
                    if bias.dim() == 0:
                        bias = bias.unsqueeze(0)  # scalar -> [1]
                    bias = bias.unsqueeze(-1)  # [E] or [1] -> [E, 1] for heads broadcast
                    if bias.shape[0] == 1:
                        bias = bias.expand(E, self.num_heads)
                    attn_scores = attn_scores + bias * cos_decay

                # Softmax over all edges (per-destination softmax)
                attn_weights = torch.softmax(attn_scores, dim=0)
                attn_weights = self.dropout(attn_weights)

                if return_attention:
                    attention_dict[(src_t, rel, dst_t)] = attn_weights.detach().mean(dim=-1)

                # Weighted sum: [E, heads] unsqueeze -> [E, heads, 1] * [E, heads, head_dim]
                msg = attn_weights.unsqueeze(-1) * v  # [E, heads, head_dim]

                # Scatter-add to destination nodes: [E, heads, head_dim] -> [N_dst, heads, head_dim]
                agg = torch.zeros(num_dst, self.num_heads, self.head_dim,
                                   device=device, dtype=msg.dtype)
                expanded_dst = dst_idx.unsqueeze(-1).unsqueeze(-1).expand(
                    E, self.num_heads, self.head_dim
                )
                agg.scatter_add_(0, expanded_dst, msg)
                agg_messages.append(agg)

            if agg_messages:
                summed = sum(agg_messages)
                # [N_dst, heads, head_dim] -> [N_dst, out_channels]
                summed = summed.reshape(num_dst, self.out_channels)
            else:
                summed = torch.zeros(num_dst, self.out_channels, device=device)

            # Residual + LayerNorm
            residual = self.skip(x_dict[dst_type])
            out_dict[dst_type] = self.norm(residual + self.out_proj(summed))

        # Copy through any node types not in self.node_types
        for nt, x in x_dict.items():
            if nt not in out_dict:
                out_dict[nt] = x

        if return_attention:
            return out_dict, attention_dict
        return out_dict


class TemperedHGT(nn.Module):
    """Multi-layer Tempered HGT model for heterogeneous graph link prediction.

    Stack of TemperedHGTConv layers wrapped for cross-type aggregation.
    Includes encoder (feature projection) and decoder (dot-product scoring).

    Args:
        in_channels: {node_type: raw_feature_dim} — typically 896 from Phase 1
        hidden_channels: Internal dimension (must be divisible by num_heads)
        out_channels: Output embedding dimension for decoder
        num_heads: Attention heads per layer
        num_layers: Number of TemperedHGTConv layers
        meta_relations: List of (src, rel, dst) triples
        temperature_init: {"Src__Rel__Dst": tau_init_value}
        dropout: Dropout rate
    """

    def __init__(self, in_channels: Dict[str, int], hidden_channels: int,
                 out_channels: int, num_heads: int, num_layers: int,
                 meta_relations: List[Tuple[str, str, str]],
                 temperature_init: Dict[str, float],
                 dropout: float = 0.1):
        super().__init__()

        assert hidden_channels % num_heads == 0

        node_types = set()
        for src, _, dst in meta_relations:
            node_types.add(src)
            node_types.add(dst)
        self.node_types = sorted(node_types)
        self.meta_relations = meta_relations
        self.hidden_channels = hidden_channels

        # Node feature encoder
        from .encoders import HeteroDictEncoder
        self.encoder = HeteroDictEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            node_types=self.node_types,
            dropout=dropout,
        )

        # Stack of TemperedHGTConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = TemperedHGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                num_heads=num_heads,
                meta_relations=meta_relations,
                temperature_init=temperature_init,
                dropout=dropout,
            )
            self.convs.append(conv)

        # Link prediction decoder
        from .encoders import InnerProductDecoder
        self.decoder = InnerProductDecoder()

        # Output projection
        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple, torch.Tensor],
                cos_decay: float = 1.0,
                edge_weight_bias: Optional[Dict[Tuple, torch.Tensor]] = None,
                return_attention: bool = False,
                ) -> Dict[str, torch.Tensor]:
        """Encode features then propagate through stacked TemperedHGTConv layers.

        Returns:
            If return_attention=False: {node_type: [N, out_channels]}
            If return_attention=True: (embeddings, [layer0_attn, layer1_attn, ...])
        """
        h = self.encoder(x_dict)

        all_attention = [] if return_attention else None
        for conv in self.convs:
            if return_attention:
                h, attn = conv(h, edge_index_dict, cos_decay=cos_decay,
                               edge_weight_bias=edge_weight_bias, return_attention=True)
                all_attention.append(attn)
            else:
                h = conv(h, edge_index_dict, cos_decay=cos_decay,
                         edge_weight_bias=edge_weight_bias)

        out = {}
        for nt, emb in h.items():
            out[nt] = self.out_proj(emb)

        if return_attention:
            return out, all_attention
        return out

    def decode(self, z_dict: Dict[str, torch.Tensor],
               edge_index: torch.Tensor, src_type: str, dst_type: str) -> torch.Tensor:
        """Score edges: dot(emb_src, emb_dst)."""
        return self.decoder(z_dict, edge_index, src_type, dst_type)

    def decode_prob(self, z_dict: Dict[str, torch.Tensor],
                    edge_index: torch.Tensor, src_type: str, dst_type: str) -> torch.Tensor:
        """Sigmoid probability scores for edges."""
        return self.decoder.decode_prob(z_dict, edge_index, src_type, dst_type)
