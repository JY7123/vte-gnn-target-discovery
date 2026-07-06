# tests/test_aggregate_results.py
import pytest
from validation.aggregate_results import ResultAggregator


class TestResultAggregator:
    @pytest.fixture
    def sample_results(self):
        return {
            "Tempered HGT": {
                "best_epoch": 42, "best_val_mrr": 0.85,
                "history": [
                    {"epoch": 0, "loss": 1.5, "val_auroc": 0.65, "val_mrr": 0.30, "val_hits@10": 0.25},
                    {"epoch": 42, "loss": 0.3, "val_auroc": 0.95, "val_mrr": 0.85, "val_hits@10": 0.80},
                ],
            },
            "Pure HGT": {
                "best_epoch": 38, "best_val_mrr": 0.72,
                "history": [
                    {"epoch": 0, "loss": 1.6, "val_auroc": 0.60, "val_mrr": 0.25, "val_hits@10": 0.20},
                    {"epoch": 38, "loss": 0.5, "val_auroc": 0.85, "val_mrr": 0.72, "val_hits@10": 0.65},
                ],
            },
            "RGCN": {
                "best_epoch": 55, "best_val_mrr": 0.68,
                "history": [
                    {"epoch": 0, "loss": 1.8, "val_auroc": 0.55, "val_mrr": 0.20, "val_hits@10": 0.15},
                    {"epoch": 55, "loss": 0.6, "val_auroc": 0.80, "val_mrr": 0.68, "val_hits@10": 0.60},
                ],
            },
        }

    def test_build_comparison_table(self, sample_results):
        agg = ResultAggregator()
        table = agg.build_comparison_table(sample_results)
        assert "Model" in table.columns
        assert "AUROC" in table.columns
        assert "MRR" in table.columns
        assert "Hits@10" in table.columns
        assert len(table) == 3

    def test_latex_export(self, sample_results):
        agg = ResultAggregator()
        latex = agg.to_latex(sample_results, caption="Model Comparison")
        assert r"\begin{table}" in latex
        assert "Tempered HGT" in latex
        assert "Pure HGT" in latex
        assert r"\end{table}" in latex

    def test_ablation_delta_positive(self, sample_results):
        agg = ResultAggregator()
        delta = agg.compute_ablation_delta(
            sample_results["Tempered HGT"],
            sample_results["Pure HGT"],
            metric="val_mrr",
        )
        assert delta > 0  # Tempered HGT outperforms Pure HGT

    def test_error_correction_summary(self):
        agg = ResultAggregator()
        ec_results = {
            "Tempered HGT": {"mean_fp_score": 0.3, "max_fp_score": 0.5},
            "Pure HGT": {"mean_fp_score": 0.7, "max_fp_score": 0.9},
        }
        summary = agg.error_correction_summary(ec_results)
        assert "suppression_ratio" in summary
        assert summary["suppression_ratio"] > 0
        assert summary["best_suppressor"] == "Tempered HGT"
