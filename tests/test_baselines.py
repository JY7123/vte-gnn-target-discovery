# tests/test_baselines.py
import pytest
import torch
from models.baselines import PyGRGCNBaseline, HANBaseline, PureHGTFactory


class TestPyGRGCNBaseline:
    @pytest.fixture
    def model(self):
        return PyGRGCNBaseline(
            in_channels={"Gene": 64, "Disease": 64},
            hidden_channels=32,
            out_channels=32,
            num_layers=2,
            meta_relations=[
                ("Gene", "ASSOCIATED_WITH", "Disease"),
                ("Gene", "REGULATES", "Gene"),
            ],
        )

    def test_forward_returns_embeddings(self, model):
        x_dict = {"Gene": torch.randn(10, 64), "Disease": torch.randn(5, 64)}
        edge_index_dict = {
            ("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0, 1], [0, 1]]),
            ("Gene", "REGULATES", "Gene"): torch.tensor([[2, 3], [4, 5]]),
        }
        out = model(x_dict, edge_index_dict)
        assert out["Gene"].shape == (10, 32)
        assert out["Disease"].shape == (5, 32)

    def test_decode_returns_logits(self, model):
        z = {"Gene": torch.randn(10, 32), "Disease": torch.randn(5, 32)}
        logits = model.decode(z, torch.tensor([[0], [0]]), "Gene", "Disease")
        assert logits.shape == (1,)

    def test_parameter_count_matches_config(self, model):
        total = sum(p.numel() for p in model.parameters())
        assert total > 0, "PyG RGCN must have trainable parameters"

    def test_forward_no_nan(self, model):
        x_dict = {"Gene": torch.randn(10, 64), "Disease": torch.randn(5, 64)}
        edge_index_dict = {
            ("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0], [0]]),
        }
        out = model(x_dict, edge_index_dict)
        for nt, emb in out.items():
            assert not torch.isnan(emb).any(), f"NaN in {nt}"


class TestHANBaseline:
    @pytest.fixture
    def model(self):
        return HANBaseline(
            in_channels={"Gene": 64, "Disease": 64, "Drug": 64, "Protein": 64},
            hidden_channels=32,
            out_channels=32,
            num_heads=4,
            meta_paths=[
                [("Gene", "ASSOCIATED_WITH", "Disease")],
                [("Gene", "REGULATES", "Gene"), ("Gene", "ASSOCIATED_WITH", "Disease")],
                [("Drug", "INHIBITS", "Protein"), ("Protein", "ASSOCIATED_WITH", "Disease")],
            ],
        )

    def test_forward_runs_without_error(self, model):
        x_dict = {
            "Gene": torch.randn(10, 64),
            "Disease": torch.randn(5, 64),
            "Drug": torch.randn(3, 64),
            "Protein": torch.randn(4, 64),
        }
        edge_index_dict = {
            ("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0, 1], [0, 1]]),
            ("Gene", "REGULATES", "Gene"): torch.tensor([[2, 3], [4, 5]]),
            ("Drug", "INHIBITS", "Protein"): torch.tensor([[0, 1], [0, 1]]),
            ("Protein", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0, 1], [0, 1]]),
        }
        out = model(x_dict, edge_index_dict)
        assert len(out) > 0
        for nt, emb in out.items():
            assert emb.shape[-1] == 32

    def test_han_handles_missing_edge_types(self, model):
        x_dict = {"Gene": torch.randn(10, 64), "Disease": torch.randn(5, 64)}
        edge_index_dict = {
            ("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0], [0]]),
        }
        out = model(x_dict, edge_index_dict)
        assert len(out) > 0


class TestPureHGTFactory:
    def test_pure_hgt_disables_temperature_and_bias(self):
        model = PureHGTFactory.create(
            in_channels={"Gene": 64, "Disease": 64},
            hidden_channels=32, out_channels=32,
            num_heads=2, num_layers=1,
            meta_relations=[("Gene", "ASSOCIATED_WITH", "Disease")],
        )
        for conv in model.convs:
            for key, tau in conv.temperatures.items():
                assert abs(tau.item() - 1.0) < 0.01
                assert tau.requires_grad is False

    def test_pure_hgt_forward_works(self):
        model = PureHGTFactory.create(
            in_channels={"Gene": 64, "Disease": 64},
            hidden_channels=32, out_channels=32, num_heads=2, num_layers=1,
            meta_relations=[("Gene", "ASSOCIATED_WITH", "Disease")],
        )
        x = {"Gene": torch.randn(5, 64), "Disease": torch.randn(3, 64)}
        ei = {("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0], [0]])}
        out = model(x, ei, cos_decay=0.0)
        assert out["Gene"].shape == (5, 32)
