# validation/aggregate_results.py
"""Aggregate baseline comparison results and export publication-ready tables."""
import pandas as pd
from typing import Dict


class ResultAggregator:
    """Build comparison tables and LaTeX exports from FairTrainer results."""

    def _best_metrics(self, result: dict) -> dict:
        """Retrieve the history entry matching best_epoch by epoch value."""
        best_epoch = result["best_epoch"]
        for entry in result["history"]:
            if entry.get("epoch") == best_epoch:
                return entry
        return result["history"][-1]  # fallback to last entry

    def build_comparison_table(self, results: Dict[str, dict]) -> pd.DataFrame:
        """Build DataFrame comparing all models on key metrics."""
        rows = []
        for model_name, result in results.items():
            best_metrics = self._best_metrics(result)
            rows.append({
                "Model": model_name,
                "AUROC": f"{best_metrics.get('val_auroc', 0):.4f}",
                "MRR": f"{best_metrics.get('val_mrr', 0):.4f}",
                "Hits@10": f"{best_metrics.get('val_hits@10', 0):.4f}",
                "Best Epoch": result["best_epoch"],
            })
        return pd.DataFrame(rows)

    def to_latex(self, results: Dict[str, dict],
                 caption: str = "Link prediction performance comparison",
                 label: str = "tab:model_comparison") -> str:
        """Export results as publication-ready LaTeX table."""
        df = self.build_comparison_table(results)
        return df.to_latex(index=False, caption=caption, label=label,
                           column_format="lccc")

    def compute_ablation_delta(self, full_result: dict, ablated_result: dict,
                                metric: str = "val_mrr") -> float:
        """Compute performance delta: full - ablated. Positive = prior helps."""
        full_val = self._best_metrics(full_result).get(metric, 0)
        ablated_val = self._best_metrics(ablated_result).get(metric, 0)
        return full_val - ablated_val

    def error_correction_summary(self, ec_results: Dict[str, dict]) -> dict:
        """Summarize error correction test.

        suppression_ratio > 0 means the best prior model suppresses false positives.
        """
        means = {name: r["mean_fp_score"] for name, r in ec_results.items()}
        max_mean = max(means.values())
        min_mean = min(means.values())

        return {
            "suppression_ratio": (max_mean - min_mean) / max_mean if max_mean > 0 else 0.0,
            "model_scores": means,
            "best_suppressor": min(means, key=means.get),
        }
