# tests/test_metrics.py
import pytest
import torch
from training.metrics import compute_auroc, compute_mrr, compute_hits_at_k


class TestMetrics:
    def test_auroc_perfect_separation(self):
        pos = torch.tensor([10.0, 9.0, 8.0])
        neg = torch.tensor([-1.0, -2.0, -3.0])
        auroc = compute_auroc(pos, neg)
        assert auroc == 1.0

    def test_auroc_random(self):
        torch.manual_seed(42)
        pos = torch.randn(100)
        neg = torch.randn(100)
        auroc = compute_auroc(pos, neg)
        assert 0.3 < auroc < 0.7

    def test_auroc_handles_empty_input(self):
        assert compute_auroc(torch.tensor([]), torch.tensor([1.0, 2.0])) == 0.5
        assert compute_auroc(torch.tensor([1.0]), torch.tensor([])) == 0.5

    def test_mrr_pos_ranked_first(self):
        pos = torch.tensor([10.0])
        neg = torch.tensor([1.0, 2.0, 3.0])
        mrr = compute_mrr(pos, neg)
        assert mrr == 1.0

    def test_mrr_pos_ranked_third(self):
        pos = torch.tensor([5.0])
        neg = torch.tensor([10.0, 8.0, 3.0, 1.0])
        mrr = compute_mrr(pos, neg)
        assert abs(mrr - 1.0/3) < 0.01

    def test_mrr_handles_empty_pos(self):
        assert compute_mrr(torch.tensor([]), torch.tensor([1.0])) == 0.0

    def test_hits_at_1_perfect(self):
        pos = torch.tensor([[10.0], [9.0]])
        neg = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
        hits1 = compute_hits_at_k(pos, neg, k=1)
        assert hits1 == 1.0

    def test_hits_at_k_zero_when_ranked_last(self):
        pos = torch.tensor([[1.0]])
        neg = torch.tensor([[10.0, 9.0, 8.0]])
        hits1 = compute_hits_at_k(pos, neg, k=1)
        assert hits1 == 0.0

    def test_hits_at_k_with_1d_inputs(self):
        pos = torch.tensor([5.0, 3.0])
        neg = torch.tensor([1.0, 2.0, 8.0, 9.0])
        hits2 = compute_hits_at_k(pos, neg, k=2)
        assert 0.0 <= hits2 <= 1.0
