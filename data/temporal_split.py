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
