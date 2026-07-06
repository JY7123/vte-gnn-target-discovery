"""Extract explanation subgraphs as JSON/CSV for Cytoscape 300 DPI rendering."""
import json, csv
from pathlib import Path
from typing import Dict, List, Tuple


class SubgraphExtractor:
    def to_json(self, explanation: dict, edge_src_dst: Dict[Tuple, List[Tuple]],
                output_path: str, threshold: float = 0.1) -> dict:
        edge_mask = explanation.get("edge_mask", {})
        nodes, edges = {}, []
        for et, masks in edge_mask.items():
            src_t, rel, dst_t = et
            names = edge_src_dst.get(et, [])
            masks_list = masks.tolist() if hasattr(masks, 'tolist') else list(masks)
            for i, (src_name, dst_name) in enumerate(names[:len(masks_list)]):
                weight = float(masks_list[i])
                if weight < threshold:
                    continue
                nodes[src_name] = src_t
                nodes[dst_name] = dst_t
                edges.append({"source": src_name, "target": dst_name, "weight": round(weight, 4), "relation": rel})
        graph = {"nodes": [{"id": n, "type": t} for n, t in nodes.items()], "edges": edges}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(graph, f, indent=2)
        return graph

    def to_csv(self, graph: dict, edges_path: str, nodes_path: str = None):
        Path(edges_path).parent.mkdir(parents=True, exist_ok=True)
        with open(edges_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "target", "weight", "relation"])
            writer.writeheader()
            for edge in graph["edges"]:
                writer.writerow(edge)
        if nodes_path:
            with open(nodes_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "type"])
                writer.writeheader()
                for node in graph["nodes"]:
                    writer.writerow(node)
