# tests/test_ablation_injection.py
import pytest
import torch
from torch_geometric.data import HeteroData
from data.ablation_injection import FalsePositiveInjector


class TestFalsePositiveInjector:
    @pytest.fixture
    def injector(self):
        import yaml, tempfile, os
        cfg = {
            "false_positive_injection": {
                "pairs": [
                    {"source": "Hmgb1", "source_type": "Gene",
                     "target": "VTE", "target_type": "Disease",
                     "relation": "ASSOCIATED_WITH"},
                ]
            }
        }
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(cfg, tmp)
        tmp.close()
        injector = FalsePositiveInjector(config_path=tmp.name)
        os.unlink(tmp.name)
        return injector

    def test_resolve_exact_match(self, injector):
        data = HeteroData()
        data["Gene"].num_nodes = 3
        data["Disease"].num_nodes = 2
        node_map = {
            "Gene": {"Hmgb1": 0, "F2": 1, "Lgals3": 2},
            "Disease": {"VTE": 0, "DVT": 1},
        }
        resolved = injector.resolve_node_indices(data, node_map)
        assert len(resolved) == 1
        et, src_idx, dst_idx = resolved[0]
        assert et == ("Gene", "ASSOCIATED_WITH", "Disease")
        assert src_idx == 0
        assert dst_idx == 0

    def test_inject_adds_edge_to_train(self, injector):
        data = HeteroData()
        data["Gene"].num_nodes = 3
        data["Disease"].num_nodes = 2
        node_map = {"Gene": {"Hmgb1": 0}, "Disease": {"VTE": 0}}
        et = ("Gene", "ASSOCIATED_WITH", "Disease")
        train_ei = {et: torch.tensor([[1], [1]])}
        neg_ei = {et: torch.tensor([[0, 2], [0, 1]])}

        new_train, new_neg, injected = injector.inject(data, node_map, train_ei, neg_ei)

        assert new_train[et].shape[1] == 2
        assert new_neg[et].shape[1] == 1
        assert len(injected) == 1
