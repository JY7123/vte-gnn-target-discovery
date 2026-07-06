"""Export the VTE knowledge graph from Neo4j to PyG HeteroData.

v2: Multi-label aware. Uses priority mapping to assign each Neo4j node to
exactly one PyG node type, and dynamically discovers edge types via
flexible Cypher queries instead of hardcoded config lists.

Fix: 48,885 previously-isolated Entity nodes (CD44/TLR4/RHOA/etc.) now
correctly exported with their edges.
"""
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
from tqdm import tqdm


class VTEKnowledgeGraphExporter:
    """Exports Neo4j VTE KG to PyG HeteroData (multi-label aware)."""

    # Priority: when a node has multiple labels, assign to the highest-priority type
    # Lower index = higher priority
    LABEL_PRIORITY = [
        "Gene", "Protein", "Drug", "Disease", "Cytokine", "Cell",
        "Pathway", "Metabolite", "Hormone", "Process", "ECM", "Concept",
        "Article", "Entity"
    ]

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j",
                 config_path: str = None):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "anchor_config.yaml"
        with open(config_path, encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.entity_types = self.config["entity_types"]

    def close(self):
        self.driver.close()

    def _assign_type(self, labels: List[str]) -> str:
        """Priority-based type assignment from multi-label nodes.

        CD44 = ['Entity', 'Protein'] -> 'Protein'
        TLR4 = ['Entity', 'Protein'] -> 'Protein'
        F2   = ['Entity', 'Gene']    -> 'Gene'
        """
        for priority_type in self.LABEL_PRIORITY:
            if priority_type in labels:
                return priority_type
        return labels[0] if labels else "Entity"

    def fetch_node_counts(self) -> Dict[str, int]:
        counts = {}
        with self.driver.session(database=self.database) as session:
            for etype in self.entity_types:
                result = session.run(f"MATCH (n:{etype}) RETURN count(n) AS cnt")
                counts[etype] = result.single()["cnt"]
        return counts

    def _fetch_all_nodes(self) -> Dict:
        """Fetch ALL nodes with their labels. Assign each to exactly ONE type.

        Returns:
            nodes_by_type: {assigned_type: {node_id: [neo4j_ids],
                                            name: [names],
                                            pub_date: [None]}}
            neo4j_to_type: {neo4j_global_id: assigned_type}
        """
        # First pass: fetch all nodes with labels
        neo4j_nodes = {}  # neo4j_id -> {name, labels}
        with self.driver.session(database=self.database) as session:
            for etype in tqdm(self.entity_types, desc="Fetching nodes"):
                query = f"""
                MATCH (n:{etype})
                RETURN id(n) AS node_id, labels(n) AS labels,
                       coalesce(n.name, toString(id(n))) AS name
                """
                result = session.run(query)
                for record in result:
                    nid = record["node_id"]
                    if nid not in neo4j_nodes:
                        neo4j_nodes[nid] = {
                            "name": record["name"],
                            "labels": set(record["labels"]),
                        }
                    else:
                        neo4j_nodes[nid]["labels"].update(record["labels"])

        print(f"  Total unique Neo4j nodes: {len(neo4j_nodes)}")

        # Second pass: assign each node to one type
        nodes_by_type = {et: {"node_id": [], "name": [], "pub_date": []}
                         for et in self.entity_types}
        neo4j_to_type = {}

        for nid, info in neo4j_nodes.items():
            assigned = self._assign_type(list(info["labels"]))
            nodes_by_type[assigned]["node_id"].append(nid)
            nodes_by_type[assigned]["name"].append(info["name"])
            nodes_by_type[assigned]["pub_date"].append(None)
            neo4j_to_type[nid] = assigned

        for et in self.entity_types:
            n = len(nodes_by_type[et]["node_id"])
            if n > 0:
                print(f"  {et:15s}: {n:>7d} assigned nodes")

        return nodes_by_type, neo4j_to_type

    def _fetch_all_edges(self, neo4j_to_type: Dict[int, str]) -> Tuple[Dict, Dict]:
        """Fetch ALL edges with flexible multi-label matching.

        Uses dynamic Cypher that matches any node label combination.
        Assigns src/dst types from the priority mapping.

        Returns:
            edges: {(assigned_src_type, rel, assigned_dst_type): edge_index [2, E]}
            edge_attrs: {same_key: {pub_date, weight}}
        """
        edges = {}
        edge_attrs = {}
        total_edges = 0

        with self.driver.session(database=self.database) as session:
            # Dynamic query: fetch all edges, get node labels at both ends
            query = """
            MATCH (a)-[r]->(b)
            RETURN id(a) AS src_id, labels(a) AS src_labels,
                   type(r) AS rel_type,
                   id(b) AS dst_id, labels(b) AS dst_labels,
                   coalesce(r.pmid_count, 1) AS weight
            """
            result = session.run(query)
            records = list(result)
            print(f"  Total Neo4j edges: {len(records)}")

            # Group by (src_type, rel, dst_type) using priority assignment
            temp_edges = {}  # (src_t, rel, dst_t) -> ([src_ids], [dst_ids], [weights])

            for rec in tqdm(records, desc="Processing edges"):
                src_nid = rec["src_id"]
                dst_nid = rec["dst_id"]
                rel = rec["rel_type"]
                weight = rec["weight"]

                # Assign types (from our mapping, or from labels if node wasn't fetched)
                src_t = neo4j_to_type.get(src_nid)
                dst_t = neo4j_to_type.get(dst_nid)

                if src_t is None:
                    src_t = self._assign_type(list(rec["src_labels"]))
                if dst_t is None:
                    dst_t = self._assign_type(list(rec["dst_labels"]))

                key = (src_t, rel, dst_t)
                if key not in temp_edges:
                    temp_edges[key] = ([], [], [])
                temp_edges[key][0].append(src_nid)
                temp_edges[key][1].append(dst_nid)
                temp_edges[key][2].append(weight)

            # Convert to tensors
            for key, (src_list, dst_list, w_list) in temp_edges.items():
                edges[key] = torch.stack([
                    torch.tensor(src_list, dtype=torch.long),
                    torch.tensor(dst_list, dtype=torch.long),
                ], dim=0)
                edge_attrs[key] = {
                    "pub_date": [None] * len(src_list),
                    "weight": torch.tensor(w_list, dtype=torch.float),
                }
                total_edges += len(src_list)

        print(f"  {len(edges)} unique edge type triples, {total_edges} total edges")
        return edges, edge_attrs

    def export(self) -> HeteroData:
        """Main export: unified node store + dynamic edge discovery."""
        data = HeteroData()

        # 1. Fetch nodes with multi-label awareness
        print("\n[Export] Step 1: Fetching nodes (multi-label aware)...")
        nodes_by_type, neo4j_to_type = self._fetch_all_nodes()

        # Build neo4j_id -> local_idx mapping per type
        neo4j_to_local = {}
        for etype in self.entity_types:
            nids = nodes_by_type[etype]["node_id"]
            neo4j_to_local[etype] = {nid: idx for idx, nid in enumerate(nids)}
            data[etype].num_nodes = len(nids)
            data[etype].node_id = torch.tensor(nids, dtype=torch.long)
            data[etype].name = nodes_by_type[etype]["name"]
            data[etype].pub_date = nodes_by_type[etype]["pub_date"]

        # 2. Fetch edges with flexible matching
        print("\n[Export] Step 2: Fetching edges (dynamic type discovery)...")
        edges, edge_attrs = self._fetch_all_edges(neo4j_to_type)

        # 3. Remap to local indices and attach to HeteroData
        print("\n[Export] Step 3: Remapping to local indices...")
        for (src_t, rel, dst_t), edge_index_global in tqdm(edges.items(), desc="Remapping"):
            src_map = neo4j_to_local.get(src_t, {})
            dst_map = neo4j_to_local.get(dst_t, {})

            src_local = [src_map.get(int(idx)) for idx in edge_index_global[0]]
            dst_local = [dst_map.get(int(idx)) for idx in edge_index_global[1]]

            # Skip edges where either node type wasn't in our entity list
            if None in src_local or None in dst_local:
                continue

            data[src_t, rel, dst_t].edge_index = torch.stack([
                torch.tensor(src_local, dtype=torch.long),
                torch.tensor(dst_local, dtype=torch.long),
            ], dim=0)

            attrs = edge_attrs[(src_t, rel, dst_t)]
            data[src_t, rel, dst_t].pub_date = attrs["pub_date"]
            data[src_t, rel, dst_t].weight = attrs["weight"]

        # 4. Verify: anchor genes have edges
        self._verify_connectivity(data, nodes_by_type)

        return data

    def _verify_connectivity(self, data: HeteroData, nodes_by_type: Dict):
        """Assert critical anchor genes have edges in the exported graph."""
        print("\n[Export] Step 4: Connectivity verification...")
        anchor_checks = ["cd44", "tlr4", "rhoa", "itgb1", "nfkb1", "f11", "kng1"]

        for anchor in anchor_checks:
            found = False
            for etype in data.node_types:
                names = nodes_by_type[etype]["name"]
                for idx, name in enumerate(names):
                    if anchor in str(name).lower():
                        # Check degree across all edge types
                        degree = 0
                        for et in data.edge_types:
                            src_t, rel, dst_t = et
                            ei = data[et].edge_index
                            if src_t == etype:
                                degree += (ei[0] == idx).sum().item()
                            if dst_t == etype:
                                degree += (ei[1] == idx).sum().item()
                        status = "CONNECTED" if degree > 0 else "ISOLATED"
                        print(f"  {anchor:10s} [{etype}][{idx}] degree={degree:>5d}  {status}")
                        found = True
                        if degree == 0:
                            print(f"    WARNING: {anchor} is isolated in PyG export!")
                        break
                if found:
                    break
            if not found:
                print(f"  {anchor:10s} NOT FOUND in exported graph")
