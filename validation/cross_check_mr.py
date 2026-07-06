"""MR causal target cross-validation for GNN link predictions.

Integrates Project 2 multi-omics Mendelian Randomization results (F11, KNG1, LRP4)
with Project 3 GNN predicted targets.
"""
import json
from typing import Dict, List


class MRCrossValidator:
    """Cross-validate GNN predictions against MR causal targets."""

    def __init__(self, mr_data_path: str = "data/mr_targets.json"):
        with open(mr_data_path, encoding="utf-8") as f:
            data = json.load(f)
        self.mr_targets = data["targets"]
        self.falsified = data.get("falsified_targets", [])
        self.mr_gene_set = {t["gene"] for t in self.mr_targets}

    def is_mr_target(self, gene: str) -> bool:
        return gene in self.mr_gene_set

    def get_mr_info(self, gene: str) -> dict:
        for t in self.mr_targets:
            if t["gene"] == gene:
                return t
        return {}

    def compute_overlap(self, gnn_predictions: List[dict]) -> dict:
        gnn_genes = {p["gene"] for p in gnn_predictions}
        intersection = gnn_genes & self.mr_gene_set
        mr_only = self.mr_gene_set - gnn_genes
        gnn_only = gnn_genes - self.mr_gene_set

        return {
            "mr_only": len(mr_only),
            "gnn_only": len(gnn_only),
            "intersection": len(intersection),
            "intersection_genes": sorted(intersection),
            "mr_only_genes": sorted(mr_only),
            "gnn_only_genes": sorted(gnn_only),
            "overlap_ratio": len(intersection) / max(len(self.mr_gene_set), 1),
        }

    def build_venn_data(self, gnn_predictions: List[dict]) -> dict:
        gnn_genes = {p["gene"] for p in gnn_predictions}
        return {
            "gnn_set": sorted(gnn_genes),
            "mr_set": sorted(self.mr_gene_set),
            "intersection": sorted(gnn_genes & self.mr_gene_set),
            "mr_targets_detail": [self.get_mr_info(g) for g in (gnn_genes & self.mr_gene_set)],
        }
