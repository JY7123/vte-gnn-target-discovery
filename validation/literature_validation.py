"""Temporal-locked literature validation. Only 2025-07 to 2026-06 = prospective."""
from datetime import datetime
from typing import Dict, List
from data.pmid_date_lookup import PMIDDateLookup


class LiteratureValidator:
    def __init__(self, train_cutoff="2024-12-31", prospective_start="2025-07-01",
                 prospective_end="2026-06-30"):
        self.train_cutoff = self._parse(train_cutoff)
        self.prospective_start = self._parse(prospective_start)
        self.prospective_end = self._parse(prospective_end)
        self.pmid_lookup = PMIDDateLookup()

    @staticmethod
    def _parse(s: str) -> datetime:
        return datetime.strptime(s[:10], "%Y-%m-%d")

    def classify_date(self, date_str: str) -> str:
        d = self._parse(date_str)
        if d <= self.train_cutoff:
            return "train_era"
        elif d <= datetime(2025, 6, 30):
            return "val_era"
        elif self.prospective_start <= d <= self.prospective_end:
            return "prospective"
        return "post_window"

    def filter_prospective(self, pmid_date_map: Dict[str, str]) -> Dict[str, str]:
        return {p: d for p, d in pmid_date_map.items() if self.classify_date(d) == "prospective"}

    def validate_predictions(self, predictions: List[dict]) -> dict:
        result = {"total_predictions": len(predictions), "prospective_hits": 0,
                  "reproduction_hits": 0, "results": []}
        for pred in predictions:
            result["results"].append({
                "gene": pred.get("gene", "unknown"),
                "score": pred.get("score", 0.0),
                "literature_status": "unvalidated",
            })
        return result
