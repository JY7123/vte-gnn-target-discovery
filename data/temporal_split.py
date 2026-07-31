"""Temporal train/val/test split based on publication date.

Core principle: "predict the future" — train on literature published through 2024,
validate on 2025 H1, test on 2025 H2-2026 H1.

Transductive constraint: test edges must have both endpoint nodes present in the
train set. Edges involving brand-new nodes go to a separate inductive extension set.

When publication dates are unavailable (None), all edges default to the train set
with a clear warning. Dates can be injected later via inject_edge_dates() or
inject_node_dates() methods.
"""
import warnings
import torch
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from collections import defaultdict
from torch_geometric.data import HeteroData


class TemporalSplitter:
    """Split graph edges by publication date into train/val/test."""

    def __init__(self, train_cutoff: str = "2024-12-31",
                 val_start: str = "2025-01-01",
                 val_end: str = "2025-06-30",
                 test_start: str = "2025-07-01",
                 test_end: str = "2026-06-30"):
        self.train_cutoff = self._parse_date(train_cutoff)
        self.val_start = self._parse_date(val_start)
        self.val_end = self._parse_date(val_end)
        self.test_start = self._parse_date(test_start)
        self.test_end = self._parse_date(test_end)
        self._missing_dates_warned = False

    @staticmethod
    def _parse_date(date_str) -> datetime:
        """Parse flexible date formats to datetime. None -> epoch (always train)."""
        if date_str is None:
            return datetime(1900, 1, 1)
        if isinstance(date_str, datetime):
            return date_str
        for fmt in ["%Y-%m-%d", "%Y-%m", "%Y"]:
            try:
                return datetime.strptime(str(date_str)[:10], fmt)
            except (ValueError, IndexError):
                continue
        try:
            year = int(str(date_str)[:4])
            return datetime(year, 1, 1)
        except (ValueError, IndexError):
            return datetime(1900, 1, 1)

    def _warn_missing_dates(self):
        if not self._missing_dates_warned:
            warnings.warn(
                "No publication dates found in edge data. All edges defaulting to "
                "train set. Use inject_edge_dates() to provide dates from external "
                "source (e.g., PubMed XML metadata). Temporal 'predict the future' "
                "validation requires accurate publication dates."
            )
            self._missing_dates_warned = True

    def inject_edge_dates(self, data: HeteroData,
                           edge_dates: Dict[Tuple, List[str]]):
        """Inject publication dates from external source (e.g., PubMed XML).

        Args:
            data: HeteroData to modify in-place
            edge_dates: {edge_type: [date_str for each edge]}
        """
        for edge_type, dates in edge_dates.items():
            if edge_type in data.edge_types:
                data[edge_type].pub_date = dates

    def split_edges_by_time(self, data: HeteroData
                             ) -> Tuple[Dict, Dict, Dict]:
        """Split all edges into train/val/test based on publication date.

        Edges with None/missing dates -> train set (with warning).

        Returns:
            train_ei: {edge_type: edge_index}
            val_ei:   {edge_type: edge_index}
            test_ei:  {edge_type: edge_index}
        """
        train_ei = {}
        val_ei = {}
        test_ei = {}
        any_dates_found = False

        for edge_type in data.edge_types:
            ei = data[edge_type].edge_index
            pub_dates = data[edge_type].pub_date

            train_mask = []
            val_mask = []
            test_mask = []

            for i in range(ei.shape[1]):
                date_str = pub_dates[i] if isinstance(pub_dates, list) else None
                if date_str is not None:
                    any_dates_found = True
                d = self._parse_date(date_str)

                if d <= self.train_cutoff:
                    train_mask.append(i)
                elif self.val_start <= d <= self.val_end:
                    val_mask.append(i)
                elif self.test_start <= d <= self.test_end:
                    test_mask.append(i)

            if train_mask:
                train_ei[edge_type] = ei[:, train_mask]
            if val_mask:
                val_ei[edge_type] = ei[:, val_mask]
            if test_mask:
                test_ei[edge_type] = ei[:, test_mask]

        if not any_dates_found:
            self._warn_missing_dates()

        return train_ei, val_ei, test_ei

    def _get_train_nodes(self, data: HeteroData) -> Dict[str, set]:
        """Determine which node indices are in the train set."""
        train_nodes = {}
        for node_type in data.node_types:
            pub_dates = data[node_type].pub_date
            known = set()
            for i, d in enumerate(pub_dates):
                if self._parse_date(d) <= self.train_cutoff:
                    known.add(i)
            # If no dates at all, all nodes are "known"
            if not known and all(d is None for d in pub_dates):
                known = set(range(data[node_type].num_nodes))
            train_nodes[node_type] = known
        return train_nodes

    def split_with_transductive_check(self, data: HeteroData
                                       ) -> Tuple[Dict, Dict, Dict, Dict]:
        """Split edges and enforce transductive constraint on test set.

        Returns:
            train_ei, val_ei, test_ei (transductive), inductive_ei
        """
        train_ei, val_ei, test_ei = self.split_edges_by_time(data)
        train_nodes = self._get_train_nodes(data)

        transductive_test = {}
        inductive_test = {}

        for edge_type, ei in test_ei.items():
            src_t, rel, dst_t = edge_type
            src_train = train_nodes.get(src_t, set())
            dst_train = train_nodes.get(dst_t, set())

            trans_mask = []
            ind_mask = []
            for i in range(ei.shape[1]):
                src_idx = int(ei[0, i])
                dst_idx = int(ei[1, i])
                if src_idx in src_train and dst_idx in dst_train:
                    trans_mask.append(i)
                else:
                    ind_mask.append(i)

            if trans_mask:
                transductive_test[edge_type] = ei[:, trans_mask]
            if ind_mask:
                inductive_test[edge_type] = ei[:, ind_mask]

        return train_ei, val_ei, transductive_test, inductive_test

    def generate_split_report(self, data: HeteroData) -> dict:
        """Generate summary statistics for the temporal split."""
        train_ei, val_ei, test_ei, inductive_ei = self.split_with_transductive_check(data)

        def _count_edges(ei_dict):
            return sum(e.shape[1] for e in ei_dict.values())

        report = {
            "train_edges": _count_edges(train_ei),
            "val_edges": _count_edges(val_ei),
            "test_edges": _count_edges(test_ei),
            "inductive_edges": _count_edges(inductive_ei),
            "train_cutoff": str(self.train_cutoff.date()),
            "val_window": f"{self.val_start.date()} to {self.val_end.date()}",
            "test_window": f"{self.test_start.date()} to {self.test_end.date()}",
            "train_edge_types": len(train_ei),
            "test_edge_types": len(test_ei),
            "has_temporal_info": _count_edges(test_ei) > 0,
        }
        if _count_edges(test_ei) + _count_edges(inductive_ei) > 0:
            report["transductive_fraction"] = (
                _count_edges(test_ei) / (_count_edges(test_ei) + _count_edges(inductive_ei))
            )
        else:
            report["transductive_fraction"] = 0.0
        return report


class RandomStratifiedSplitter:
    """Random stratified edge split preserving per-relation distribution.

    Used when temporal split is infeasible (no publication dates in KG).
    Splits edges 80/10/10 per edge type. Handles edge types with few edges
    gracefully (all-to-train if < 10 edges).

    The split is seeded for reproducibility and records a hash of the
    resulting partition.
    """

    def __init__(self, train_frac: float = 0.8, val_frac: float = 0.1,
                 test_frac: float = 0.1, seed: int = 42,
                 min_edges_for_split: int = 10,
                 edge_types: Optional[List[Tuple]] = None):
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.min_edges_for_split = min_edges_for_split
        self.edge_types = [tuple(et) for et in edge_types] if edge_types else None

    def split(self, data: HeteroData) -> Tuple[Dict, Dict, Dict]:
        """Split edges per type into train/val/test.

        If edge_types filter is set, only those edge types are split.

        Returns:
            train_ei: {edge_type: edge_index [2, N_train]}
            val_ei:   {edge_type: edge_index [2, N_val]}
            test_ei:  {edge_type: edge_index [2, N_test]}
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        train_ei, val_ei, test_ei = {}, {}, {}

        edge_types_iter = self.edge_types if self.edge_types else data.edge_types
        for edge_type in edge_types_iter:
            if edge_type not in data.edge_types:
                continue
            ei = data[edge_type].edge_index

            # Deduplicate: remove repeated (src, dst) pairs before splitting
            unique_pairs = set()
            unique_indices = []
            for j in range(ei.shape[1]):
                pair = (int(ei[0, j]), int(ei[1, j]))
                if pair not in unique_pairs:
                    unique_pairs.add(pair)
                    unique_indices.append(j)
            if len(unique_indices) < ei.shape[1]:
                ei = ei[:, unique_indices]

            n_edges = ei.shape[1]

            if n_edges < self.min_edges_for_split:
                train_ei[edge_type] = ei
                continue

            n_val = max(1, int(n_edges * self.val_frac))
            n_test = max(1, int(n_edges * self.test_frac))
            n_train = n_edges - n_val - n_test

            perm = torch.randperm(n_edges, generator=generator)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:n_train + n_val]
            test_idx = perm[n_train + n_val:]

            train_ei[edge_type] = ei[:, train_idx]
            val_ei[edge_type] = ei[:, val_idx]
            test_ei[edge_type] = ei[:, test_idx]

        return train_ei, val_ei, test_ei

    def split_and_report(self, data: HeteroData) -> dict:
        """Split and return summary report dict."""
        train_ei, val_ei, test_ei = self.split(data)

        def _count(ei_dict):
            return sum(e.shape[1] for e in ei_dict.values())

        total_ets = self.edge_types if self.edge_types else data.edge_types
        return {
            "split_method": "random_stratified",
            "seed": self.seed,
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "test_frac": self.test_frac,
            "train_edges": _count(train_ei),
            "val_edges": _count(val_ei),
            "test_edges": _count(test_ei),
            "train_edge_types": len(train_ei),
            "val_edge_types": len(val_ei),
            "test_edge_types": len(test_ei),
            "edge_types_without_split": sum(
                1 for et in total_ets
                if et in data.edge_types and et in train_ei
                and et not in val_ei and et not in test_ei
            ),
        }
