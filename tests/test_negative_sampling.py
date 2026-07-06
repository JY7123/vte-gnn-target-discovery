# tests/test_negative_sampling.py
import pytest
import torch
from data.negative_sampling import (
    DegreePreservingNegativeSampler,
    HardNegativeSampler,
    NegativeSamplingPipeline,
)


class TestDegreePreservingNegativeSampler:
    @pytest.fixture
    def sampler(self):
        return DegreePreservingNegativeSampler(seed=42)

    @pytest.fixture
    def sample_edge_index(self):
        # 5 nodes, edges: 0->1, 0->2, 0->3, 4->1
        return {("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([
            [0, 0, 0, 4], [1, 2, 3, 1]
        ])}

    def test_degree_preserving_sampling_respects_node_degrees(self, sampler, sample_edge_index):
        """High-degree nodes should appear more often in negative samples."""
        pos_edges = sample_edge_index
        num_nodes_dict = {"Gene": 5, "Disease": 4}

        negatives = sampler.sample(
            pos_edges, num_nodes_dict, num_negatives_per_edge=2
        )

        key = ("Gene", "ASSOCIATED_WITH", "Disease")
        neg_src = negatives[key][0]
        neg_dst = negatives[key][1]

        # Node 0 (degree 3) should appear more than Node 2 (degree 1) as source
        src_counts = torch.bincount(neg_src, minlength=5)
        # Node 1 (degree 2) should appear more than Node 3 (degree 1) as destination
        dst_counts = torch.bincount(neg_dst, minlength=4)

        # At minimum, high-degree nodes get sampled
        assert src_counts[0] > 0  # degree 3 node is in negatives
        assert dst_counts[1] > 0  # degree 2 node is in negatives

    def test_negative_edges_are_not_in_positive_set(self, sampler, sample_edge_index):
        """No negative sample should duplicate a real edge."""
        pos_edges = sample_edge_index
        num_nodes_dict = {"Gene": 5, "Disease": 4}

        negatives = sampler.sample(
            pos_edges, num_nodes_dict, num_negatives_per_edge=10
        )

        key = ("Gene", "ASSOCIATED_WITH", "Disease")
        pos_set = set(zip(
            pos_edges[key][0].tolist(),
            pos_edges[key][1].tolist()
        ))
        neg_set = set(zip(
            negatives[key][0].tolist(),
            negatives[key][1].tolist()
        ))

        assert pos_set.isdisjoint(neg_set)

    def test_reproducibility_with_seed(self):
        """Same seed produces identical negative samples."""
        ei = {("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0, 1], [2, 3]])}
        num_nodes = {"Gene": 4, "Disease": 4}

        s1 = DegreePreservingNegativeSampler(seed=42)
        s2 = DegreePreservingNegativeSampler(seed=42)

        n1 = s1.sample(ei, num_nodes, num_negatives_per_edge=2)
        n2 = s2.sample(ei, num_nodes, num_negatives_per_edge=2)
        key = ("Gene", "ASSOCIATED_WITH", "Disease")
        assert torch.equal(n1[key], n2[key])


class TestHardNegativeSampler:
    @pytest.fixture
    def config(self):
        return {
            "falsified_targets": ["Padi4", "Hmgb1"],
            "cross_tissue_negatives": [
                ["CNS_specific_proteins", "Venous_thrombosis"],
            ]
        }

    def test_falsified_targets_generate_negative_edges(self, config):
        """Padi4 and Hmgb1 (Project 1 falsified targets) must be negative sources."""
        sampler = HardNegativeSampler(config, seed=42)

        node_name_to_idx = {
            "Gene": {"Padi4": 0, "Hmgb1": 1, "F2": 2},
            "Disease": {"VTE": 0, "Deep_vein_thrombosis": 1},
        }

        hard_negatives = sampler.sample_falsified_target_negatives(
            node_name_to_idx,
            num_negatives_per_target=3
        )

        assert ("Gene", "ASSOCIATED_WITH", "Disease") in hard_negatives
        neg_edges = hard_negatives[("Gene", "ASSOCIATED_WITH", "Disease")]
        # Padi4 (idx 0) must be in source nodes
        assert 0 in neg_edges[0].tolist()


class TestNegativeSamplingPipeline:
    def test_pipeline_combines_degree_and_hard_negatives(self):
        """Pipeline output = degree-preserving negatives + hard negatives."""
        config = {
            "falsified_targets": ["Padi4", "Hmgb1"],
            "cross_tissue_negatives": [],
        }
        pipeline = NegativeSamplingPipeline(config, seed=42)

        pos_edges = {
            ("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0, 1], [2, 3]])
        }
        num_nodes = {"Gene": 5, "Disease": 4}
        node_names = {
            "Gene": {0: "Padi4", 1: "Hmgb1", 2: "F2", 3: "F11", 4: "Lgals3"},
            "Disease": {0: "VTE", 1: "DVT", 2: "PE", 3: "Thrombosis"},
        }

        negatives = pipeline.generate(
            pos_edges, num_nodes, node_names,
            num_negatives_per_edge=2
        )

        key = ("Gene", "ASSOCIATED_WITH", "Disease")
        assert key in negatives
        assert negatives[key].shape[0] == 2  # [2, num_negatives]
        assert negatives[key].shape[1] > 0

    def test_total_negative_count_matches_requested(self):
        """num_negatives_per_edge * num_pos_edges total negatives per edge type."""
        pipeline = NegativeSamplingPipeline({}, seed=42)
        pos_edges = {
            ("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0, 1], [0, 2]])
        }
        num_nodes = {"Gene": 3, "Disease": 3}
        node_names = {"Gene": {0: "A", 1: "B", 2: "C"}, "Disease": {0: "X", 1: "Y", 2: "Z"}}

        negatives = pipeline.generate(pos_edges, num_nodes, node_names,
                                       num_negatives_per_edge=3)
        key = ("Gene", "ASSOCIATED_WITH", "Disease")
        # 2 positive edges * 3 negatives per edge = 6 expected
        assert negatives[key].shape[1] >= 6
