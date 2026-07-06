import pytest
from validation.literature_validation import LiteratureValidator


class TestLiteratureValidator:
    @pytest.fixture
    def validator(self):
        return LiteratureValidator()

    def test_classify_date(self, validator):
        assert validator.classify_date("2018-03-15") == "train_era"
        assert validator.classify_date("2025-01-15") == "val_era"
        assert validator.classify_date("2025-08-01") == "prospective"
        assert validator.classify_date("2026-01-20") == "prospective"

    def test_filter_prospective(self, validator):
        pmids = {"12345": "2025-08-15", "67890": "2023-06-01", "11111": "2026-01-10"}
        prospective = validator.filter_prospective(pmids)
        assert len(prospective) == 2
        assert "12345" in prospective
        assert "67890" not in prospective

    def test_validate_predictions(self, validator):
        preds = [{"gene": "F2", "score": 0.95}, {"gene": "NEW", "score": 0.85}]
        r = validator.validate_predictions(preds)
        assert r["total_predictions"] == 2
