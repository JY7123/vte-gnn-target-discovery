# tests/test_temporal_split.py
import pytest
import torch
from torch_geometric.data import HeteroData
from data.temporal_split import TemporalSplitter


def make_mock_heterodata() -> HeteroData:
    """Create a minimal HeteroData fixture with known temporal labels."""
    data = HeteroData()

    # 4 Gene nodes with pub_dates
    data["Gene"].num_nodes = 4
    data["Gene"].node_id = torch.tensor([101, 102, 103, 104])
    data["Gene"].name = ["FUT8", "Lgals3", "Padi4", "F11"]
    data["Gene"].pub_date = ["2018-03-15", "2019-07-01", "2025-02-10", "2020-11-20"]

    # 3 Disease nodes
    data["Disease"].num_nodes = 3
    data["Disease"].node_id = torch.tensor([201, 202, 203])
    data["Disease"].name = ["VTE", "DVT", "PE"]
    data["Disease"].pub_date = ["2017-01-01", "2018-06-15", "2019-03-20"]

    # Edges with publication dates
    data["Gene", "ASSOCIATED_WITH", "Disease"].edge_index = torch.tensor([
        [0, 1, 2, 3],  # FUT8, Lgals3, Padi4, F11
        [0, 1, 0, 2]   # VTE, DVT, VTE, PE
    ])
    data["Gene", "ASSOCIATED_WITH", "Disease"].pub_date = [
        "2019-08-01",   # FUT8-VTE: train (<=2024)
        "2020-02-15",   # Lgals3-DVT: train (<=2024)
        "2025-08-01",   # Padi4-VTE: test (2025 H2)
        "2021-05-20",   # F11-PE: train (<=2024)
    ]

    return data


class TestTemporalSplitter:
    @pytest.fixture
    def splitter(self):
        return TemporalSplitter(
            train_cutoff="2024-12-31",
            val_start="2025-01-01",
            val_end="2025-06-30",
            test_start="2025-07-01",
            test_end="2026-06-30",
        )

    def test_edge_temporal_split_assigns_correct_splits(self, splitter):
        """Edges must be split by their publication date into train/val/test."""
        data = make_mock_heterodata()
        train_ei, val_ei, test_ei = splitter.split_edges_by_time(data)

        et = ("Gene", "ASSOCIATED_WITH", "Disease")

        # Train edges: FUT8-VTE (2019), Lgals3-DVT (2020), F11-PE (2021)
        train_pairs = set(zip(train_ei[et][0].tolist(), train_ei[et][1].tolist()))
        assert (0, 0) in train_pairs  # FUT8-VTE
        assert (1, 1) in train_pairs  # Lgals3-DVT
        assert (3, 2) in train_pairs  # F11-PE
        assert (2, 0) not in train_pairs  # Padi4-VTE is test

        # Test edges: Padi4-VTE (2025-03-10)
        test_pairs = set(zip(test_ei[et][0].tolist(), test_ei[et][1].tolist()))
        assert (2, 0) in test_pairs

    def test_transductive_constraint_enforcement(self, splitter):
        """Test edges must have BOTH endpoint nodes in the train set."""
        data = make_mock_heterodata()
        data["Gene", "REGULATES", "Gene"].edge_index = torch.tensor([[2, 3], [0, 1]])
        data["Gene", "REGULATES", "Gene"].pub_date = [
            "2025-08-01",
            "2025-08-01",
        ]

        _, _, test_ei, inductive_ei = splitter.split_with_transductive_check(data)

        et = ("Gene", "REGULATES", "Gene")
        # Padi4 (idx 2) has pub_date 2025-02-10 which is AFTER train_cutoff 2024-12-31,
        # so node 2 is NOT in train_nodes. Node 0 IS in train_nodes.
        # Edge 2->0: src=2 (not in train) -> inductive
        # Edge 3->1: src=3 (2020-11-20, in train), dst=1 (2019-07-01, in train) -> transductive
        if et in inductive_ei:
            ind_pairs = set(zip(inductive_ei[et][0].tolist(), inductive_ei[et][1].tolist()))
            assert (2, 0) in ind_pairs

        if et in test_ei:
            trans_pairs = set(zip(test_ei[et][0].tolist(), test_ei[et][1].tolist()))
            assert (3, 1) in trans_pairs

        # At least one of these edge types should be present
        assert et in test_ei or et in inductive_ei

    def test_split_report_generates_statistics(self, splitter):
        """The split should produce a statistics report."""
        data = make_mock_heterodata()
        report = splitter.generate_split_report(data)
        assert "train_edges" in report
        assert "val_edges" in report
        assert "test_edges" in report
        assert isinstance(report["train_edges"], int)

    def test_handles_missing_dates_gracefully(self, splitter):
        """When pub_date is None, edges should default to train set with warning."""
        data = HeteroData()
        data["Gene"].num_nodes = 3
        data["Gene"].node_id = torch.tensor([1, 2, 3])
        data["Gene"].name = ["A", "B", "C"]
        data["Gene"].pub_date = [None, None, None]

        data["Disease"].num_nodes = 2
        data["Disease"].node_id = torch.tensor([10, 20])
        data["Disease"].name = ["X", "Y"]
        data["Disease"].pub_date = [None, None]

        data["Gene", "ASSOCIATED_WITH", "Disease"].edge_index = torch.tensor([
            [0, 1], [0, 1]
        ])
        data["Gene", "ASSOCIATED_WITH", "Disease"].pub_date = [None, None]

        train_ei, val_ei, test_ei = splitter.split_edges_by_time(data)
        et = ("Gene", "ASSOCIATED_WITH", "Disease")
        # All edges should default to train
        assert et in train_ei
        assert train_ei[et].shape[1] == 2
        # Val and test should be empty (or not contain this edge type)
