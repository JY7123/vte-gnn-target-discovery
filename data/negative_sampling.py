"""Negative sampling for Link Prediction with hard negatives.

Strategy:
1. Degree-preserving random negatives (baseline)
2. Hard negatives from falsified targets (Padi4, Hmgb1 from Project 1)
3. Cross-tissue/cross-compartment forced negatives
"""
import torch
import random
from typing import Dict, Tuple, List, Optional


class DegreePreservingNegativeSampler:
    """Sample negative edges with probability proportional to node degree.

    Samples source nodes proportional to their degree in the positive edge set,
    then for each source picks a destination (degree-weighted) that is not
    already connected in the positive edges.  This avoids the rejection-sampling
    trap where high-degree sources connect to most destinations.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def _compute_degree_weights(self, edge_index: torch.Tensor, num_nodes: int,
                                 is_source: bool) -> torch.Tensor:
        """Compute sampling weights proportional to node degree."""
        idx = 0 if is_source else 1
        degrees = torch.bincount(edge_index[idx], minlength=num_nodes).float()
        degrees = degrees + 1.0  # allow zero-degree nodes to be sampled
        return degrees / degrees.sum()

    def _build_pos_adjacency(self, edge_index: torch.Tensor
                              ) -> List[set]:
        """Build per-source set of positive destinations for O(1) look-up."""
        adj = [set() for _ in range(edge_index[0].max().item() + 1)]
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            adj[s].add(d)
        # pad to num_src in case some source indices have no edges
        num_src = edge_index[0].max().item() + 1
        return adj

    def sample(self, pos_edge_dict: Dict[Tuple, torch.Tensor],
               num_nodes_dict: Dict[str, int],
               num_negatives_per_edge: int = 1
               ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """Generate degree-preserving negative edges.

        Returns:
            {edge_type_tuple: negative_edge_index [2, total_negatives]}
        """
        negatives = {}

        for edge_type, pos_ei in pos_edge_dict.items():
            src_t, rel, dst_t = edge_type
            num_src = num_nodes_dict[src_t]
            num_dst = num_nodes_dict[dst_t]
            num_pos = pos_ei.shape[1]
            total_needed = num_pos * num_negatives_per_edge

            src_weights = self._compute_degree_weights(pos_ei, num_src, is_source=True)
            dst_weights = self._compute_degree_weights(pos_ei, num_dst, is_source=False)
            pos_adj = self._build_pos_adjacency(pos_ei)

            # Build set of already-sampled negative pairs for this round
            neg_src_list: List[int] = []
            neg_dst_list: List[int] = []
            seen_negatives: set = set()

            # Pre-compute available destinations for each source
            all_dst = set(range(num_dst))

            attempts = 0
            max_attempts = total_needed * 50

            while len(neg_src_list) < total_needed and attempts < max_attempts:
                src = torch.multinomial(src_weights, 1, generator=self._rng).item()
                attempts += 1

                # Available destinations: not in positive set for this source
                if src < len(pos_adj):
                    forbidden = pos_adj[src]
                else:
                    forbidden = set()
                available = list(all_dst - forbidden)
                if not available:
                    continue

                # Weighted sample from available destinations
                avail_weights = dst_weights[available]
                # Re-normalise the sub-distribution
                if avail_weights.sum() == 0:
                    avail_weights = torch.ones(len(available)) / len(available)
                else:
                    avail_weights = avail_weights / avail_weights.sum()

                dst_idx = torch.multinomial(
                    avail_weights, 1, generator=self._rng
                ).item()
                dst = available[dst_idx]

                if (src, dst) not in seen_negatives:
                    neg_src_list.append(src)
                    neg_dst_list.append(dst)
                    seen_negatives.add((src, dst))

            # Fallback: fill remaining slots with random edges not in positive set
            if len(neg_src_list) < total_needed:
                remaining = total_needed - len(neg_src_list)
                for _ in range(remaining):
                    for _ in range(100):  # inline attempt loop
                        s = torch.randint(0, num_src, (1,),
                                          generator=self._rng).item()
                        d = torch.randint(0, num_dst, (1,),
                                          generator=self._rng).item()
                        pos_forbidden = pos_adj[s] if s < len(pos_adj) else set()
                        if d not in pos_forbidden and (s, d) not in seen_negatives:
                            neg_src_list.append(s)
                            neg_dst_list.append(d)
                            seen_negatives.add((s, d))
                            break

            negatives[edge_type] = torch.tensor(
                [neg_src_list[:total_needed], neg_dst_list[:total_needed]],
                dtype=torch.long
            )

        return negatives


class HardNegativeSampler:
    """Generate hard negative edges from domain knowledge constraints.

    Accepts node mapping in either format:
      - name -> idx:  {"Gene": {"Padi4": 0, "Hmgb1": 1}}
      - idx  -> name: {"Gene": {0: "Padi4", 1: "Hmgb1"}}
    """

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed = seed
        random.seed(seed)

    @staticmethod
    def _detect_format(node_map: Dict[str, Dict]
                       ) -> str:
        """Return 'name_to_idx' or 'idx_to_name' by inspecting first key type."""
        for type_map in node_map.values():
            if type_map:
                first_key = next(iter(type_map.keys()))
                return "name_to_idx" if isinstance(first_key, str) else "idx_to_name"
        return "idx_to_name"

    def _resolve_name_to_idx(self, node_map: Dict[str, Dict],
                              entity_type: str, name: str) -> Optional[int]:
        """Resolve an entity name to its local node index.

        Handles both {name: idx} and {idx: name} formats.
        """
        type_map = node_map.get(entity_type, {})
        if not type_map:
            return None

        fmt = self._detect_format(node_map)
        if fmt == "name_to_idx":
            return type_map.get(name)
        else:
            # idx_to_name: reverse lookup
            for idx, node_name in type_map.items():
                if node_name == name:
                    return idx
        return None

    def _get_node_indices(self, node_map: Dict[str, Dict],
                           entity_type: str) -> List[int]:
        """Return all node indices for a given entity type.

        Handles both {name: idx} and {idx: name} formats.
        """
        type_map = node_map.get(entity_type, {})
        if not type_map:
            return []

        fmt = self._detect_format(node_map)
        if fmt == "name_to_idx":
            return list(type_map.values())
        else:
            return list(type_map.keys())

    def sample_falsified_target_negatives(self, node_map: Dict[str, Dict],
                                           num_negatives_per_target: int = 5
                                           ) -> Dict[Tuple, torch.Tensor]:
        """Create negative edges between falsified gene targets and VTE/Disease nodes."""
        negatives = {}
        falsified = self.config.get("falsified_targets", [])

        disease_indices = self._get_node_indices(node_map, "Disease")
        if not disease_indices or not falsified:
            return negatives

        neg_src = []
        neg_dst = []
        for target_name in falsified:
            src_idx = self._resolve_name_to_idx(node_map, "Gene", target_name)
            if src_idx is None:
                continue
            for dst_idx in disease_indices:
                for _ in range(num_negatives_per_target):
                    neg_src.append(src_idx)
                    neg_dst.append(dst_idx)

        if neg_src:
            negatives[("Gene", "ASSOCIATED_WITH", "Disease")] = torch.tensor(
                [neg_src, neg_dst], dtype=torch.long
            )

        return negatives

    def sample_cross_tissue_negatives(self, node_map: Dict[str, Dict],
                                       num_per_pair: int = 10
                                       ) -> Dict[Tuple, torch.Tensor]:
        """Generate cross-compartment negatives."""
        negatives = {}
        cross_tissue = self.config.get("cross_tissue_negatives", [])

        for tissue_type, disease_name in cross_tissue:
            protein_indices = self._get_node_indices(node_map, "Protein")
            disease_idx = self._resolve_name_to_idx(node_map, "Disease", disease_name)
            if disease_idx is None or not protein_indices:
                continue

            neg_src = []
            neg_dst = []
            sampled = random.sample(protein_indices,
                                     min(num_per_pair, len(protein_indices)))
            for protein_idx in sampled:
                neg_src.append(protein_idx)
                neg_dst.append(disease_idx)

            if neg_src:
                negatives[("Protein", "ASSOCIATED_WITH", "Disease")] = torch.tensor(
                    [neg_src, neg_dst], dtype=torch.long
                )

        return negatives


class NegativeSamplingPipeline:
    """Combine degree-preserving and hard negative sampling."""

    def __init__(self, config: dict, seed: int = 42):
        self.degree_sampler = DegreePreservingNegativeSampler(seed=seed)
        self.hard_sampler = HardNegativeSampler(config, seed=seed)

    def generate(self, pos_edge_dict: Dict[Tuple, torch.Tensor],
                 num_nodes_dict: Dict[str, int],
                 node_map: Dict[str, Dict],
                 num_negatives_per_edge: int = 1
                 ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """Generate combined negative edges.

        Args:
            pos_edge_dict: {edge_type: edge_index [2, num_edges]}
            num_nodes_dict: {entity_type: num_nodes}
            node_map: {entity_type: {local_idx: node_name}} or
                      {entity_type: {node_name: local_idx}}
                      (the HardNegativeSampler handles both formats)
            num_negatives_per_edge: number of negatives per positive edge

        Returns:
            {edge_type: negative_edge_index [2, total_negatives]}
        """
        # 1. Degree-preserving negatives
        degree_negs = self.degree_sampler.sample(
            pos_edge_dict, num_nodes_dict, num_negatives_per_edge
        )

        # 2. Hard negatives
        hard_negs = {}
        hard_negs.update(
            self.hard_sampler.sample_falsified_target_negatives(node_map)
        )
        hard_negs.update(
            self.hard_sampler.sample_cross_tissue_negatives(node_map)
        )

        # 3. Merge
        merged = {}
        all_edge_types = set(degree_negs.keys()) | set(hard_negs.keys())
        for et in all_edge_types:
            deg = degree_negs.get(et, torch.empty((2, 0), dtype=torch.long))
            hard = hard_negs.get(et, torch.empty((2, 0), dtype=torch.long))
            if deg.shape[1] == 0 and hard.shape[1] == 0:
                continue
            parts = [p for p in [deg, hard] if p.shape[1] > 0]
            merged[et] = torch.cat(parts, dim=1)

        return merged
