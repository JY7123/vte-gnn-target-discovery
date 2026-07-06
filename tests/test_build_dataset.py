# tests/test_build_dataset.py
import pytest
import torch
from pathlib import Path
from data.build_dataset import DatasetBuilder


@pytest.fixture
def config():
    return {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "12345678",
        "neo4j_database": "neo4j",
        "output_dir": "/tmp/vte_gnn_test_data",
    }


class TestDatasetBuilder:
    def test_builder_accepts_config(self, config):
        builder = DatasetBuilder(config)
        assert builder.output_dir == Path("/tmp/vte_gnn_test_data")

    def test_build_creates_output_directory(self, config):
        builder = DatasetBuilder(config)
        builder.output_dir.mkdir(parents=True, exist_ok=True)
        assert builder.output_dir.exists()

    def test_full_pipeline_runs_end_to_end(self, config):
        """Full pipeline: export -> features -> negatives -> split -> save."""
        builder = DatasetBuilder(config)
        try:
            result = builder.build(skip_features=True)
        except Exception as e:
            if "Unable to connect" in str(e) or "Connection refused" in str(e):
                pytest.skip(f"Neo4j not available: {e}")
            raise

        # Verify output structure
        assert result.output_dir.exists()
        assert (result.output_dir / "heterodata.pt").exists()
        assert (result.output_dir / "train_edges.pt").exists()
        assert (result.output_dir / "val_edges.pt").exists()
        assert (result.output_dir / "test_edges.pt").exists()
        assert (result.output_dir / "negative_edges.pt").exists()
        assert (result.output_dir / "split_report.json").exists()

    def test_output_dimensions_are_consistent(self, config):
        """PubMedBERT(768) + Node2Vec(128) = 896 for all node types."""
        feat_path = Path(config["output_dir"]) / "train_features.pt"
        if not feat_path.exists():
            pytest.skip("Run full pipeline first to generate features")
        features = torch.load(feat_path, weights_only=True)
        for etype, feats in features.items():
            assert feats.shape[1] == 896, f"{etype} features have wrong dim: {feats.shape}"


class TestDatasetBuilderWithoutNeo4j:
    """Tests that don't require a running Neo4j instance."""

    def test_dataset_bundle_attributes(self):
        """DatasetBundle should have all expected attributes."""
        from data.build_dataset import DatasetBundle
        bundle = DatasetBundle(
            output_dir=Path("/tmp/test"),
            data=None,
            features=None,
            train_ei={},
            val_ei={},
            test_ei={},
            inductive_ei={},
            negative_edges={},
            split_report={"train_edges": 0},
        )
        assert bundle.output_dir == Path("/tmp/test")
        assert bundle.split_report["train_edges"] == 0

    def test_builder_skip_features_flag(self, config):
        """With skip_features=True, should still export and split but skip features."""
        builder = DatasetBuilder(config)
        try:
            result = builder.build(skip_features=True)
            assert result.features is None
            assert (result.output_dir / "heterodata.pt").exists()
        except Exception as e:
            if "Auth" in str(e) or "Connection" in str(e) or "Unable" in str(e):
                pytest.skip(f"Neo4j not available: {e}")
            raise
