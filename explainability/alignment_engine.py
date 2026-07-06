"""Anchor alignment scoring for GNNExplainer subgraphs."""
from typing import Dict, Set, List


class AnchorAlignmentEngine:
    def __init__(self, positive_anchors: List[str], pathway_anchors: List[str] = None,
                 negative_anchors: List[str] = None):
        self.positive_anchors = set(positive_anchors)
        self.pathway_anchors = set(pathway_anchors or [])
        self.negative_anchors = set(negative_anchors or [])
        self.all_known = self.positive_anchors | self.pathway_anchors

    def compute_anchor_alignment(self, subgraph_genes: Set[str]) -> float:
        if not subgraph_genes:
            return 0.0
        inter = len(subgraph_genes & self.all_known)
        union = len(subgraph_genes | self.all_known)
        return inter / union if union > 0 else 0.0

    def classify_target(self, explained_genes: Set[str], prediction_score: float) -> dict:
        matched = explained_genes & self.all_known
        novel = explained_genes - self.all_known
        alignment = self.compute_anchor_alignment(explained_genes)
        return {
            "type": "shared_downstream" if alignment > 0.2 else "novel_mechanism",
            "alignment_score": alignment,
            "matched_anchors": sorted(matched),
            "novel_genes": sorted(novel),
            "prediction_score": prediction_score,
        }

    def build_radar_data(self, targets: Dict[str, Set[str]]) -> dict:
        scores = {}
        for name, genes in targets.items():
            alignment = self.compute_anchor_alignment(genes)
            pathway = len(genes & self.pathway_anchors) / max(len(self.pathway_anchors), 1)
            novelty = len(genes - self.all_known) / max(len(genes), 1)
            scores[name] = [alignment, pathway, novelty]
        return {"categories": ["Anchor Alignment", "Pathway Coverage", "Novelty"], "scores": scores}
