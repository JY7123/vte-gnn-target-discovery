import pytest
from validation.cross_check_mr import MRCrossValidator


class TestMRCrossValidator:
    @pytest.fixture
    def validator(self):
        return MRCrossValidator(mr_data_path="data/mr_targets.json")

    def test_load_mr_targets(self, validator):
        assert len(validator.mr_targets) == 3
        assert validator.mr_targets[0]["gene"] == "F11"

    def test_gene_in_mr_list(self, validator):
        assert validator.is_mr_target("F11") is True
        assert validator.is_mr_target("F2") is False

    def test_compute_venn_intersection(self, validator):
        gnn_predictions = [
            {"gene": "F11", "score": 0.95},
            {"gene": "KNG1", "score": 0.82},
            {"gene": "NEW_GENE", "score": 0.78},
            {"gene": "F2", "score": 0.91},
        ]
        result = validator.compute_overlap(gnn_predictions)
        assert result["mr_only"] == 1  # LRP4
        assert result["gnn_only"] == 2  # NEW_GENE, F2
        assert result["intersection"] == 2  # F11, KNG1
        assert len(result["intersection_genes"]) == 2

    def test_venn_data_for_plotting(self, validator):
        gnn_predictions = [
            {"gene": "F11", "score": 0.95},
            {"gene": "KNG1", "score": 0.82},
        ]
        venn = validator.build_venn_data(gnn_predictions)
        assert "gnn_set" in venn
        assert "mr_set" in venn
        assert "intersection" in venn
