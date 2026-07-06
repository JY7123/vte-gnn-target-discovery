"""End-to-end dataset construction pipeline.

Orchestrates: Neo4j export -> node features -> negative sampling -> temporal split -> save.
"""
import json
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional
from torch_geometric.data import HeteroData

from .neo4j_to_pyg import VTEKnowledgeGraphExporter
from .temporal_split import TemporalSplitter


class DatasetBuilder:
    """Build the complete training-ready dataset from Neo4j KG.

    Usage:
        builder = DatasetBuilder({
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "12345678",
            "neo4j_database": "neo4j",
            "output_dir": "data/processed",
        })
        result = builder.build()
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "data/processed"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, skip_features: bool = False) -> "DatasetBundle":
        """Run the full data pipeline.

        Args:
            skip_features: If True, skip PubMedBERT+Node2Vec (for quick structural test).
        """
        print("=" * 60)
        print("Phase 1: VTE GNN Dataset Construction")
        print("=" * 60)

        # Step 1: Export from Neo4j
        print("\n[1/5] Exporting from Neo4j...")
        exporter = VTEKnowledgeGraphExporter(
            uri=self.config["neo4j_uri"],
            user=self.config["neo4j_user"],
            password=self.config["neo4j_password"],
            database=self.config.get("neo4j_database", "neo4j"),
        )
        data = exporter.export()
        total_nodes = sum(data[nt].num_nodes for nt in data.node_types)
        total_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
        print(f"  Exported: {total_nodes} nodes, {total_edges} edges")
        print(f"  Node types: {list(data.node_types)}")
        print(f"  Edge types: {len(data.edge_types)}")

        # Save raw HeteroData
        torch.save(data, self.output_dir / "heterodata.pt")

        # Step 2: Temporal split
        print("\n[2/5] Performing temporal split...")
        splitter = TemporalSplitter()
        train_ei, val_ei, test_ei, inductive_ei = splitter.split_with_transductive_check(data)

        report = splitter.generate_split_report(data)
        print(f"  Train: {report['train_edges']} edges")
        print(f"  Val:   {report['val_edges']} edges")
        print(f"  Test:  {report['test_edges']} edges (transductive)")
        print(f"  Inductive: {report['inductive_edges']} edges")
        if not report.get("has_temporal_info", False):
            print("  WARNING: No temporal info -- all edges in train set. "
                  "Use inject_edge_dates() to add publication dates.")

        with open(self.output_dir / "split_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Step 3: Node features (skip if requested)
        features = None
        if not skip_features:
            print("\n[3/5] Generating node features...")
            try:
                from .node_features import NodeFeaturePipeline, EntitySummaryBuilder

                pipeline = NodeFeaturePipeline(
                    pubmedbert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    node2vec_dim=128,
                    output_dir=str(self.output_dir),
                )

                # Build summaries from Neo4j
                summary_builder = EntitySummaryBuilder(
                    neo4j_uri=self.config["neo4j_uri"],
                    neo4j_user=self.config["neo4j_user"],
                    neo4j_password=self.config["neo4j_password"],
                    database=self.config.get("neo4j_database", "neo4j"),
                )

                node_data = {
                    nt: {
                        "name": data[nt].name,
                        "node_id": data[nt].node_id.tolist(),
                    }
                    for nt in data.node_types
                }
                summaries = summary_builder.build_all_summaries(node_data)

                pubmedbert_feats = pipeline.generate_pubmedbert_features(summaries)

                num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
                node2vec_feats = pipeline.generate_node2vec_features(train_ei, num_nodes_dict)

                features = pipeline.combine_and_normalize(pubmedbert_feats, node2vec_feats)
                pipeline.save_features(features, "train")

                print(f"  Feature dimensions: {pipeline.output_dim}")
                for nt, feats in features.items():
                    print(f"    {nt}: {feats.shape}")
            except ImportError as e:
                print(f"  WARNING: Feature generation skipped -- {e}")
        else:
            print("\n[3/5] Skipping node features (skip_features=True)")

        # Step 4: Negative sampling
        print("\n[4/5] Generating negative samples...")
        try:
            from .negative_sampling import NegativeSamplingPipeline

            # Load anchor config for negative sampling settings
            anchor_path = Path(__file__).parent.parent / "config" / "anchor_config.yaml"
            neg_config = {}
            if anchor_path.exists():
                with open(anchor_path, encoding='utf-8') as f:
                    anchor_cfg = yaml.safe_load(f)
                neg_config = anchor_cfg.get("negative_sampling", {})

            neg_pipeline = NegativeSamplingPipeline(neg_config, seed=42)

            node_name_to_idx = {}
            for nt in data.node_types:
                node_name_to_idx[nt] = {
                    name: idx for idx, name in enumerate(data[nt].name)
                }

            num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
            negative_edges = neg_pipeline.generate(
                train_ei, num_nodes_dict, node_name_to_idx,
                num_negatives_per_edge=1
            )
            total_neg = sum(ei.shape[1] for ei in negative_edges.values())
            print(f"  Total negative edges: {total_neg}")
        except ImportError as e:
            print(f"  WARNING: Negative sampling skipped -- {e}")
            negative_edges = {}

        # Step 5: Save all processed data
        print("\n[5/5] Saving processed dataset...")
        torch.save(train_ei, self.output_dir / "train_edges.pt")
        torch.save(val_ei, self.output_dir / "val_edges.pt")
        torch.save(test_ei, self.output_dir / "test_edges.pt")
        torch.save(inductive_ei, self.output_dir / "inductive_edges.pt")
        torch.save(negative_edges, self.output_dir / "negative_edges.pt")

        print(f"\nDataset saved to {self.output_dir}")
        print("=" * 60)

        exporter.close()
        return DatasetBundle(
            output_dir=self.output_dir,
            data=data,
            features=features,
            train_ei=train_ei,
            val_ei=val_ei,
            test_ei=test_ei,
            inductive_ei=inductive_ei,
            negative_edges=negative_edges,
            split_report=report,
        )


class DatasetBundle:
    """Container for all processed dataset components."""
    def __init__(self, output_dir: Path, data: HeteroData,
                 features: Optional[Dict], train_ei: Dict, val_ei: Dict,
                 test_ei: Dict, inductive_ei: Dict, negative_edges: Dict,
                 split_report: dict):
        self.output_dir = output_dir
        self.data = data
        self.features = features
        self.train_ei = train_ei
        self.val_ei = val_ei
        self.test_ei = test_ei
        self.inductive_ei = inductive_ei
        self.negative_edges = negative_edges
        self.split_report = split_report
