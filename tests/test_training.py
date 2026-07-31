# tests/test_training.py
import pytest
import torch
from pathlib import Path


class TestLinkPredictionTrainer:
    @pytest.fixture
    def tiny_model(self):
        from models.tempered_hgt import TemperedHGT
        return TemperedHGT(
            in_channels={"Gene": 64, "Disease": 64},
            hidden_channels=32,
            out_channels=32,
            num_heads=2,
            num_layers=1,
            meta_relations=[("Gene", "ASSOCIATED_WITH", "Disease")],
            temperature_init={"Gene__ASSOCIATED_WITH__Disease": 1.0},
        )

    @pytest.fixture
    def tiny_data(self):
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["Gene"].num_nodes = 20
        data["Gene"].x = torch.randn(20, 64)
        data["Disease"].num_nodes = 10
        data["Disease"].x = torch.randn(10, 64)
        data["Gene", "ASSOCIATED_WITH", "Disease"].edge_index = torch.tensor([
            [0, 1, 2, 3, 4], [0, 0, 1, 1, 2]
        ])
        return data

    def test_trainer_initializes(self, tiny_model):
        from training.link_prediction import LinkPredictionTrainer
        trainer = LinkPredictionTrainer(
            model=tiny_model, learning_rate=1e-3,
            num_epochs=5, device="cpu",
        )
        assert trainer.model is tiny_model
        assert trainer.num_epochs == 5

    def test_train_one_epoch_loss_decreases(self, tiny_model, tiny_data):
        from training.link_prediction import LinkPredictionTrainer
        trainer = LinkPredictionTrainer(
            model=tiny_model, learning_rate=5e-2,
            num_epochs=10, device="cpu",
        )

        et = ("Gene", "ASSOCIATED_WITH", "Disease")
        train_ei = {et: tiny_data[et].edge_index}
        neg_ei = {et: torch.tensor([[5, 6, 7, 8, 9], [5, 6, 7, 8, 9]])}

        # Pre-training loss
        trainer.model.eval()
        with torch.no_grad():
            z_init = tiny_model(
                {"Gene": tiny_data["Gene"].x, "Disease": tiny_data["Disease"].x},
                train_ei, cos_decay=1.0,
            )
            pos_logits_init = tiny_model.decode(z_init, train_ei[et], "Gene", "Disease")
            loss_init = trainer._bce_loss(pos_logits_init, torch.ones_like(pos_logits_init))

        # Train multiple epochs to guarantee decrease
        for ep in range(3):
            trainer.train_epoch(tiny_data, train_ei, neg_ei, epoch=ep)

        trainer.model.eval()
        with torch.no_grad():
            z_after = tiny_model(
                {"Gene": tiny_data["Gene"].x, "Disease": tiny_data["Disease"].x},
                train_ei, cos_decay=1.0,
            )
            pos_logits_after = tiny_model.decode(z_after, train_ei[et], "Gene", "Disease")
            loss_after = trainer._bce_loss(pos_logits_after, torch.ones_like(pos_logits_after))

        assert loss_after < loss_init, f"Loss did not decrease: {loss_after:.4f} >= {loss_init:.4f}"

    def test_early_stopping_triggers(self, tiny_model):
        from training.link_prediction import LinkPredictionTrainer
        trainer = LinkPredictionTrainer(
            model=tiny_model, learning_rate=1e-3,
            num_epochs=100, patience=2, device="cpu",
        )
        trainer.best_val_mrr = 1.0
        trainer.patience_counter = 2
        assert trainer._should_stop_early() is True

    def test_checkpoint_save_and_load(self, tiny_model, tmp_path):
        from training.link_prediction import LinkPredictionTrainer
        trainer = LinkPredictionTrainer(
            model=tiny_model, learning_rate=1e-3,
            num_epochs=5, device="cpu",
            checkpoint_dir=str(tmp_path),
        )
        trainer._save_checkpoint(epoch=0, val_mrr=0.8)
        ckpt_path = tmp_path / "checkpoint_epoch_0.pt"
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, weights_only=True)
        assert ckpt["epoch"] == 0
        assert abs(ckpt["val_mrr"] - 0.8) < 0.01

    def test_fit_runs_multiple_epochs(self, tiny_model, tiny_data):
        from training.link_prediction import LinkPredictionTrainer
        trainer = LinkPredictionTrainer(
            model=tiny_model, learning_rate=1e-2,
            num_epochs=5, device="cpu",
        )
        et = ("Gene", "ASSOCIATED_WITH", "Disease")
        train_ei = {et: tiny_data[et].edge_index}
        neg_ei = {et: torch.tensor([[5, 6, 7, 8, 9], [5, 6, 7, 8, 9]])}
        val_ei = {et: torch.tensor([[10], [5]])}

        result = trainer.fit(
            tiny_data,
            train_ei=train_ei,
            val_ei=val_ei,
            train_neg_ei=neg_ei,
            val_neg_ei=neg_ei,
            verbose=False,
        )
        assert "best_epoch" in result
        assert "best_val_mrr" in result
        assert len(result["history"]) > 0
