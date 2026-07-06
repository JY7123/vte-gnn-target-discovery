import pytest, json
from explainability.subgraph_extractor import SubgraphExtractor


class TestSubgraphExtractor:
    @pytest.fixture
    def extractor(self):
        return SubgraphExtractor()

    def test_to_json(self, extractor, tmp_path):
        expl = {"edge_idx": 0, "edge_mask": {("Gene", "REGULATES", "Gene"): [0.8, 0.2, 0.9, 0.05]}}
        names = {("Gene", "REGULATES", "Gene"): [("FUT8", "Lgals3"), ("F2", "Lgals3"), ("Lgals3", "CD44"), ("WEAK", "X")]}
        out = tmp_path / "test.json"
        g = extractor.to_json(expl, names, str(out), threshold=0.3)
        assert len(g["nodes"]) > 0
        assert len(g["edges"]) == 2  # WEAK filtered (0.05 < 0.3), F2->Lgals3 filtered (0.2 < 0.3)

    def test_to_csv(self, extractor, tmp_path):
        graph = {"nodes": [{"id": "A", "type": "Gene"}], "edges": [{"source": "A", "target": "B", "weight": 0.8, "relation": "REGULATES"}]}
        ep = tmp_path / "edges.csv"
        np = tmp_path / "nodes.csv"
        extractor.to_csv(graph, str(ep), str(np))
        assert ep.exists()
        assert np.exists()
