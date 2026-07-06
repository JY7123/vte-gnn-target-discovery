"""Path contradiction detection for GNN explanations."""
from typing import Dict, List, Tuple


class ContradictionGate:
    def __init__(self, falsified_targets: List[str], max_path_length: int = 3):
        self.falsified = set(t.lower() for t in falsified_targets)
        self.max_path_length = max_path_length

    def check_path(self, path_edges: Dict[Tuple, List[Tuple]],
                   gene_names: Dict[int, str]) -> dict:
        contaminated = []
        all_nodes = set()
        for et, edges in path_edges.items():
            for src, dst in edges:
                all_nodes.add(src)
                all_nodes.add(dst)
        for node_idx in all_nodes:
            name = gene_names.get(node_idx, "").lower()
            if any(f in name for f in self.falsified):
                contaminated.append(gene_names.get(node_idx, str(node_idx)))
        return {
            "contaminated": len(contaminated) > 0,
            "contaminated_nodes": contaminated,
            "contradiction_score": len(contaminated),
            "clean_path_length": len(all_nodes) - len(contaminated),
            "total_nodes": len(all_nodes),
        }

    def batch_check(self, explanations: List[dict], gene_names: Dict[int, str]) -> List[dict]:
        return [{**exp, "contradiction": self.check_path(
            exp.get("explanation_edges", {}), gene_names
        )} for exp in explanations]
