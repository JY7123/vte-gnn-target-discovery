import pytest
import torch
from explainability.gnnexplainer_vte import VTEExplainer


class TestVTEExplainer:
    @pytest.fixture
    def model_and_data(self):
        from models.tempered_hgt import TemperedHGT
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["Gene"].num_nodes = 10
        data["Gene"].x = torch.randn(10, 16)
        data["Disease"].num_nodes = 5
        data["Disease"].x = torch.randn(5, 16)
        et = ("Gene", "ASSOCIATED_WITH", "Disease")
        data[et].edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
        model = TemperedHGT(
            in_channels={"Gene": 16, "Disease": 16},
            hidden_channels=8, out_channels=8, num_heads=2, num_layers=1,
            meta_relations=[et],
            temperature_init={"Gene__ASSOCIATED_WITH__Disease": 1.0},
        )
        return model, data

    def test_model_frozen_during_explanation(self, model_and_data):
        model, data = model_and_data
        orig = {n: p.clone() for n, p in model.named_parameters()}
        explainer = VTEExplainer(model, num_epochs=30)
        explainer.explain_edge(data, ("Gene", "ASSOCIATED_WITH", "Disease"), 0,
                               {"Gene": data["Gene"].x, "Disease": data["Disease"].x})
        for n, p in model.named_parameters():
            assert torch.allclose(p, orig[n], atol=1e-5), f"{n} changed!"

    def test_explain_top_k(self, model_and_data):
        model, data = model_and_data
        explainer = VTEExplainer(model, num_epochs=50)
        results = explainer.explain_top_k(
            data, ("Gene", "ASSOCIATED_WITH", "Disease"), k=3,
            x_dict={"Gene": data["Gene"].x, "Disease": data["Disease"].x},
        )
        assert len(results) <= 3
