import pytest
from explainability.contradiction_gate import ContradictionGate


class TestContradictionGate:
    @pytest.fixture
    def gate(self):
        return ContradictionGate(falsified_targets=["Padi4", "Hmgb1"])

    def test_detect_padi4(self, gate):
        path = {("Gene", "REGULATES", "Gene"): [(0, 1), (1, 2)]}
        names = {0: "FUT8", 1: "Padi4", 2: "Lgals3"}
        assert gate.check_path(path, names)["contaminated"] is True

    def test_clean_path(self, gate):
        path = {("Gene", "REGULATES", "Gene"): [(0, 1)]}
        names = {0: "FUT8", 1: "F2"}
        assert gate.check_path(path, names)["contaminated"] is False

    def test_contradiction_score(self, gate):
        path = {("Gene", "REGULATES", "Gene"): [(0, 1), (1, 2), (2, 3)]}
        names = {0: "FUT8", 1: "Padi4", 2: "Hmgb1", 3: "Lgals3"}
        assert gate.check_path(path, names)["contradiction_score"] == 2
