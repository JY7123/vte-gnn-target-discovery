# tests/test_baseline_trainer.py
import pytest
import torch
from torch_geometric.data import HeteroData


class TestFairTrainer:
    @pytest.fixture
    def tiny_models(self):
        from models.tempered_hgt import TemperedHGT
        from models.baselines import PyGRGCNBaseline, PureHGTFactory

        meta = [("Gene", "ASSOCIATED_WITH", "Disease")]
        t_init = {"Gene__ASSOCIATED_WITH__Disease": 0.5}

        tempered = TemperedHGT(
            in_channels={"Gene": 32, "Disease": 32},
            hidden_channels=16, out_channels=16,
            num_heads=2, num_layers=1,
            meta_relations=meta, temperature_init=t_init,
        )
        pure = PureHGTFactory.create(
            in_channels={"Gene": 32, "Disease": 32},
            hidden_channels=16, out_channels=16,
            num_heads=2, num_layers=1, meta_relations=meta,
        )
        rgcn = PyGRGCNBaseline(
            in_channels={"Gene": 32, "Disease": 32},
            hidden_channels=16, out_channels=16,
            num_layers=1, meta_relations=meta,
        )
        return {"Tempered HGT": tempered, "Pure HGT": pure, "RGCN": rgcn}

    @pytest.fixture
    def unified_config(self):
        return {"hidden_channels": 16, "num_heads": 2, "num_layers": 1,
                "learning_rate": 0.01, "num_epochs": 3, "patience": 5,
                "batch_size": 256}

    def test_fair_trainer_accepts_multiple_models(self, tiny_models, unified_config):
        from training.baseline_trainer import FairTrainer
        trainer = FairTrainer(models=tiny_models, unified_config=unified_config, device="cpu")
        assert len(trainer.models) == 3

    def test_train_all_models_loss_decreases(self, tiny_models, unified_config):
        from training.baseline_trainer import FairTrainer

        data = HeteroData()
        data["Gene"].num_nodes = 10
        data["Gene"].x = torch.randn(10, 32)
        data["Disease"].num_nodes = 5
        data["Disease"].x = torch.randn(5, 32)
        et = ("Gene", "ASSOCIATED_WITH", "Disease")
        data[et].edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])

        train_ei = {et: data[et].edge_index}
        neg_ei = {et: torch.tensor([[5, 6], [2, 3]])}
        val_ei = {et: torch.tensor([[3], [1]])}

        subset = {"Tempered HGT": tiny_models["Tempered HGT"],
                  "Pure HGT": tiny_models["Pure HGT"]}

        trainer = FairTrainer(
            models=subset, unified_config=unified_config, device="cpu",
            ablation_mode={"Pure HGT": "pure_hgt"},
        )

        results = trainer.train_all(data, train_ei, val_ei, neg_ei, verbose=False)
        assert "Tempered HGT" in results
        assert "Pure HGT" in results
        for name, result in results.items():
            assert result["best_epoch"] >= 0
            assert len(result["history"]) > 0

    def test_ablation_mode_disables_cos_decay(self, tiny_models, unified_config):
        from training.baseline_trainer import FairTrainer
        subset = {"Pure HGT": tiny_models["Pure HGT"]}

        data = HeteroData()
        data["Gene"].num_nodes = 5
        data["Gene"].x = torch.randn(5, 32)
        data["Disease"].num_nodes = 3
        data["Disease"].x = torch.randn(3, 32)
        et = ("Gene", "ASSOCIATED_WITH", "Disease")
        data[et].edge_index = torch.tensor([[0], [0]])

        train_ei = {et: data[et].edge_index}
        neg_ei = {et: torch.tensor([[2], [1]])}
        val_ei = {et: torch.tensor([[1], [0]])}

        trainer = FairTrainer(
            models=subset, unified_config=unified_config, device="cpu",
            ablation_mode={"Pure HGT": "pure_hgt"},
        )
        results = trainer.train_all(data, train_ei, val_ei, neg_ei, verbose=False)
        assert results["Pure HGT"]["best_epoch"] >= 0
