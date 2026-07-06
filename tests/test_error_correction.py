# tests/test_error_correction.py
import pytest
import torch
from validation.error_correction import ErrorCorrectionTest


class TestErrorCorrectionTest:
    def test_prior_model_suppresses_false_positives(self):
        """Tempered HGT should not score false positives HIGHER than Pure HGT."""
        torch.manual_seed(42)
        from models.tempered_hgt import TemperedHGT
        from models.baselines import PureHGTFactory

        meta = [("Gene", "ASSOCIATED_WITH", "Disease")]
        t_init = {"Gene__ASSOCIATED_WITH__Disease": 4.0}

        tempered = TemperedHGT(
            in_channels={"Gene": 16, "Disease": 16},
            hidden_channels=8, out_channels=8,
            num_heads=2, num_layers=1,
            meta_relations=meta, temperature_init=t_init,
        )
        pure = PureHGTFactory.create(
            in_channels={"Gene": 16, "Disease": 16},
            hidden_channels=8, out_channels=8,
            num_heads=2, num_layers=1, meta_relations=meta,
        )

        # Copy weights so only tau differs
        pure_state = pure.state_dict()
        for key in tempered.state_dict():
            if "temperatures" not in key:
                tempered.state_dict()[key].copy_(pure_state[key])

        # Train briefly
        x = {"Gene": torch.randn(10, 16), "Disease": torch.randn(5, 16)}
        ei = {("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0, 1, 2], [0, 0, 1]])}

        for model in [tempered, pure]:
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            for _ in range(20):
                opt.zero_grad()
                cos = 1.0 if model is tempered else 0.0
                z = model(x, ei, cos_decay=cos)
                pos = model.decode(z, ei[("Gene", "ASSOCIATED_WITH", "Disease")], "Gene", "Disease")
                loss = -torch.log(torch.sigmoid(pos) + 1e-8).mean()
                loss.backward()
                opt.step()

        # Test on unseen false-positive edge
        fp_edge = torch.tensor([[5], [3]])
        with torch.no_grad():
            z_t = tempered(x, ei, cos_decay=1.0)
            z_p = pure(x, ei, cos_decay=0.0)
            score_t = tempered.decode(z_t, fp_edge, "Gene", "Disease").item()
            score_p = pure.decode(z_p, fp_edge, "Gene", "Disease").item()

        # Tempered (higher tau) should suppress false positives compared to Pure
        assert score_t <= score_p, (
            f"Tempered scored fp_edge ({score_t:.3f}) higher than Pure ({score_p:.3f})"
        )

    def test_error_correction_test_accepts_injected_edges(self):
        test = ErrorCorrectionTest(
            injected_edges=[(("Gene", "ASSOCIATED_WITH", "Disease"), 0, 1)],
            model_names=["Tempered HGT", "Pure HGT"],
        )
        assert len(test.injected_edges) == 1
        assert test.injected_edges[0][1] == 0

    def test_run_returns_valid_scores(self):
        from models.tempered_hgt import TemperedHGT
        from torch_geometric.data import HeteroData

        model = TemperedHGT(
            in_channels={"Gene": 16, "Disease": 16},
            hidden_channels=8, out_channels=8,
            num_heads=2, num_layers=1,
            meta_relations=[("Gene", "ASSOCIATED_WITH", "Disease")],
            temperature_init={"Gene__ASSOCIATED_WITH__Disease": 1.0},
        )
        data = HeteroData()
        data["Gene"].num_nodes = 5
        data["Gene"].x = torch.randn(5, 16)
        data["Disease"].num_nodes = 3
        data["Disease"].x = torch.randn(3, 16)

        test = ErrorCorrectionTest(
            injected_edges=[(("Gene", "ASSOCIATED_WITH", "Disease"), 0, 0)],
            model_names=["Test"],
        )
        ei = {("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0], [0]])}
        results = test.run({"Test": model}, data, ei, {"Test": 1.0})
        assert "Test" in results
        assert "mean_fp_score" in results["Test"]
        assert len(results["Test"]["fp_scores"]) == 1
