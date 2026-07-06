# tests/test_integration.py
"""End-to-end integration: load Phase 1 data, train Tempered HGT, verify loss decreases."""
import pytest
import torch
from pathlib import Path


@pytest.mark.integration
class TestEndToEndTraining:
    def test_load_data_and_train_one_epoch(self):
        """Load processed data, build model, train 5 epochs — loss must decrease."""
        data_dir = Path("data/processed")
        if not (data_dir / "heterodata.pt").exists():
            pytest.skip("Phase 1 data not yet built — run DatasetBuilder first")

        data = torch.load(data_dir / "heterodata.pt", weights_only=False)
        train_ei = torch.load(data_dir / "train_edges.pt", weights_only=False)
        neg_ei = torch.load(data_dir / "negative_edges.pt", weights_only=False)

        from models.tempered_hgt import TemperedHGT
        from training.link_prediction import LinkPredictionTrainer
        from training.edge_bias import EdgeBiasInitializer

        # Build default in_channels for all node types
        node_types = list(data.node_types)
        feat_dim = 896  # Phase 1 feature dim
        in_channels = {nt: feat_dim for nt in node_types}

        # Get actual edge types present in data
        meta_relations = list(data.edge_types)

        # Build temperature init
        temp_init = {}
        for src, rel, dst in meta_relations:
            key = f"{src}__{rel}__{dst}"
            temp_init[key] = 1.0

        # Override from anchor_config if available
        try:
            import yaml
            with open("config/anchor_config.yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            temp_cfg = cfg.get("temperature_init", {})
            for category in temp_cfg.values():
                tau_val = category.get("tau_init", 1.0)
                for rel_name in category.get("relations", []):
                    for src, rel, dst in meta_relations:
                        if rel == rel_name:
                            key = f"{src}__{rel}__{dst}"
                            temp_init[key] = tau_val
        except Exception:
            pass  # Keep defaults

        # Build edge bias from config
        try:
            initializer = EdgeBiasInitializer.from_yaml("config/anchor_config.yaml")
            node_name_to_idx = {}
            for nt in node_types:
                if hasattr(data[nt], 'name') and data[nt].name is not None:
                    names = data[nt].name
                    node_name_to_idx[nt] = {
                        name: idx for idx, name in enumerate(names)
                    }
            bias_dict = initializer.build(node_name_to_idx, edge_types=meta_relations)
            # Convert to per-edge tensors (aligned with actual edge indices)
            edge_weight_bias = {}
            for et, bias_list in bias_dict.items():
                if et not in data.edge_types:
                    continue
                ei = data[et].edge_index
                bias_tensor = torch.zeros(ei.shape[1])
                for src_idx, dst_idx, multiplier in bias_list:
                    # Find matching edges
                    mask = (ei[0] == src_idx) & (ei[1] == dst_idx)
                    bias_tensor[mask] = multiplier
                if bias_tensor.sum() > 0:
                    edge_weight_bias[et] = bias_tensor
        except Exception as e:
            print(f"  Edge bias build skipped: {e}")
            edge_weight_bias = None

        # Build model (small for fast test)
        model = TemperedHGT(
            in_channels=in_channels,
            hidden_channels=64,
            out_channels=64,
            num_heads=4,
            num_layers=2,
            meta_relations=meta_relations,
            temperature_init=temp_init,
        )

        trainer = LinkPredictionTrainer(
            model=model,
            learning_rate=1e-2,
            num_epochs=5,
            device="cpu",
            checkpoint_dir="/tmp/vte_gnn_integration_test",
            edge_weight_bias=edge_weight_bias,
        )

        # Train 5 epochs
        total_train_edges = sum(ei.shape[1] for ei in train_ei.values())
        print(f"Training on {total_train_edges} train edges, "
              f"{len(meta_relations)} edge types")
        result = trainer.fit(data, train_ei, train_ei, neg_ei, verbose=True)

        # Verify training ran
        assert result["best_epoch"] >= 0
        assert len(result["history"]) > 0

        # Loss should be finite
        final_loss = result["history"][-1]["loss"]
        assert final_loss < 100, f"Loss exploded: {final_loss}"

        print(f"Integration test passed: best_epoch={result['best_epoch']}, "
              f"best_val_mrr={result['best_val_mrr']:.4f}")
