import pytest
from explainability.alignment_engine import AnchorAlignmentEngine


class TestAnchorAlignmentEngine:
    @pytest.fixture
    def engine(self):
        return AnchorAlignmentEngine(
            positive_anchors=["FUT8", "Lgals3", "CD44", "ITGB1"],
            pathway_anchors=["TNF", "TLR4", "NFKB1"],
        )

    def test_compute_jaccard_perfect_overlap(self, engine):
        assert engine.compute_anchor_alignment({"FUT8", "Lgals3", "CD44"}) > 0.4

    def test_no_overlap(self, engine):
        assert engine.compute_anchor_alignment({"XYZ1", "ABC2"}) == 0.0

    def test_classify_shared_downstream(self, engine):
        r = engine.classify_target({"Lgals3", "CD44", "TNF"}, 0.8)
        assert r["type"] == "shared_downstream"
        assert r["alignment_score"] > 0

    def test_classify_novel(self, engine):
        r = engine.classify_target({"NEW1", "NEW2"}, 0.9)
        assert r["type"] == "novel_mechanism"

    def test_radar_data(self, engine):
        radar = engine.build_radar_data({"A": {"Lgals3", "CD44"}, "B": {"NEW1"}})
        assert len(radar["scores"]) == 2
