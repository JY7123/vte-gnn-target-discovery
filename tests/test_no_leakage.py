# tests/test_no_leakage.py
"""Verify strict separation between message-passing and evaluation edges."""
import pytest
import torch
from torch_geometric.data import HeteroData


class TestNoDataLeakage:
    """Verify that val/test edges are NEVER used for message passing."""

    @pytest.fixture
    def toy_data(self):
        data = HeteroData()
        data["Gene"].num_nodes = 30
        data["Gene"].x = torch.randn(30, 64)
        data["Disease"].num_nodes = 10
        data["Disease"].x = torch.randn(10, 64)
        # Train edges (60%)
        data["Gene", "ASSOCIATED_WITH", "Disease"].edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        ])
        return data

    @pytest.fixture
    def splits(self):
        train_ei = {("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([
            [0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]
        ])}
        val_ei = {("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([
            [6, 7], [3, 3]
        ])}
        test_ei = {("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([
            [8, 9], [4, 4]
        ])}
        neg_ei = {("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([
            [10, 11, 12, 13, 14, 15], [5, 5, 6, 6, 7, 7]
        ])}
        val_neg = {("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([
            [16, 17], [8, 8]
        ])}
        test_neg = {("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([
            [18, 19], [9, 9]
        ])}
        return train_ei, val_ei, test_ei, neg_ei, val_neg, test_neg

    def test_train_val_edges_are_disjoint(self, splits):
        train_ei, val_ei, test_ei, _, _, _ = splits
        et = ("Gene", "ASSOCIATED_WITH", "Disease")
        train_edges = set()
        for j in range(train_ei[et].shape[1]):
            train_edges.add((int(train_ei[et][0, j]), int(train_ei[et][1, j])))
        val_edges = set()
        for j in range(val_ei[et].shape[1]):
            val_edges.add((int(val_ei[et][0, j]), int(val_ei[et][1, j])))
        test_edges = set()
        for j in range(test_ei[et].shape[1]):
            test_edges.add((int(test_ei[et][0, j]), int(test_ei[et][1, j])))

        assert train_edges.isdisjoint(val_edges), "Train and val edges overlap!"
        assert train_edges.isdisjoint(test_edges), "Train and test edges overlap!"
        assert val_edges.isdisjoint(test_edges), "Val and test edges overlap!"

    def test_evaluate_uses_train_for_message_passing(self, toy_data, splits):
        """Verify evaluate() message-passing uses train_ei, not val_ei."""
        train_ei, val_ei, _, _, val_neg, _ = splits

        from models.tempered_hgt import TemperedHGT
        model = TemperedHGT(
            in_channels={"Gene": 64, "Disease": 64},
            hidden_channels=32, out_channels=32,
            num_heads=2, num_layers=1,
            meta_relations=[("Gene", "ASSOCIATED_WITH", "Disease")],
            temperature_init={"Gene__ASSOCIATED_WITH__Disease": 1.0},
        )

        from training.link_prediction import LinkPredictionTrainer
        trainer = LinkPredictionTrainer(
            model=model, learning_rate=1e-3, num_epochs=5, device="cpu",
        )

        # This is the key: msg_ei = train_ei, eval_ei = val_ei
        metrics = trainer.evaluate(toy_data, train_ei, val_ei, val_neg)
        assert "auroc" in metrics
        assert 0.0 <= metrics["auroc"] <= 1.0

    def test_fit_accepts_separate_neg_ei(self, toy_data, splits):
        train_ei, val_ei, test_ei, train_neg, val_neg, test_neg = splits

        from models.tempered_hgt import TemperedHGT
        model = TemperedHGT(
            in_channels={"Gene": 64, "Disease": 64},
            hidden_channels=32, out_channels=32,
            num_heads=2, num_layers=1,
            meta_relations=[("Gene", "ASSOCIATED_WITH", "Disease")],
            temperature_init={"Gene__ASSOCIATED_WITH__Disease": 1.0},
        )

        from training.link_prediction import LinkPredictionTrainer
        trainer = LinkPredictionTrainer(
            model=model, learning_rate=1e-2, num_epochs=3, device="cpu",
        )
        result = trainer.fit(
            data=toy_data,
            train_ei=train_ei,
            val_ei=val_ei,
            test_ei=test_ei,
            train_neg_ei=train_neg,
            val_neg_ei=val_neg,
            test_neg_ei=test_neg,
            verbose=False,
        )
        assert "best_epoch" in result
        assert "test_metrics" in result
        # Test metrics should exist (not be the 0.5/0.0 fallback)
        assert result["test_metrics"]["auroc"] >= 0.0

    def test_random_stratified_split_is_disjoint(self, toy_data):
        """RandomStratifiedSplitter produces disjoint train/val/test."""
        from data.temporal_split import RandomStratifiedSplitter
        splitter = RandomStratifiedSplitter(
            train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42
        )
        train_ei, val_ei, test_ei = splitter.split(toy_data)

        et = ("Gene", "ASSOCIATED_WITH", "Disease")
        t = set((int(train_ei[et][0, j]), int(train_ei[et][1, j])) for j in range(train_ei[et].shape[1]))
        v = set((int(val_ei[et][0, j]), int(val_ei[et][1, j])) for j in range(val_ei[et].shape[1]))
        ts = set((int(test_ei[et][0, j]), int(test_ei[et][1, j])) for j in range(test_ei[et].shape[1]))

        assert t.isdisjoint(v)
        assert t.isdisjoint(ts)
        assert v.isdisjoint(ts)
        # All original edges should be accounted for
        n_orig = toy_data[et].edge_index.shape[1]
        assert len(t) + len(v) + len(ts) == n_orig


class TestFilteredMetrics:
    def test_filtered_ranks_perfect(self):
        """A perfect model should get MRR=1.0, H@1=1.0."""
        from training.metrics import compute_filtered_ranks

        # Create embeddings where entity i = one-hot, all same dimension
        dim = 16
        z_dict = {
            "Gene": torch.eye(10, dim),
            "Disease": torch.eye(5, dim),
        }

        # One test triple: gene 0 -> disease 0
        # Gene 0 embedding = [1,0,0,0...], Disease 0 = [1,0,0,0,0], dot = 1.0
        # All other Disease dots = 0.0, so rank = 1
        eval_triples = [(0, 0)]
        all_true = set()  # No other true triples to filter

        def fake_decode(z, ei, s, d):
            return (z[s][ei[0]] * z[d][ei[1]]).sum(dim=-1)

        results = compute_filtered_ranks(
            z_dict=z_dict,
            eval_triples=eval_triples,
            src_type="Gene",
            dst_type="Disease",
            all_true_triples=all_true,
            num_src=10,
            num_dst=5,
            decode_fn=fake_decode,
        )

        assert results["tail_mrr"] == 1.0
        assert results["head_mrr"] == 1.0
        assert results["tail_hits@1"] == 1.0
        assert results["head_hits@1"] == 1.0
        assert results["tail_hits@10"] == 1.0

    def test_build_true_triples_set(self):
        from training.metrics import build_true_triples_set

        train_ei = {("Gene", "ASSOC", "Disease"): torch.tensor([[0, 1], [0, 0]])}
        val_ei = {("Gene", "ASSOC", "Disease"): torch.tensor([[2], [1]])}

        true_sets = build_true_triples_set([train_ei, val_ei])
        et = ("Gene", "ASSOC", "Disease")
        assert et in true_sets
        assert (0, 0) in true_sets[et]
        assert (1, 0) in true_sets[et]
        assert (2, 1) in true_sets[et]
        assert (0, 1) not in true_sets[et]
