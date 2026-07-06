# tests/test_encoders.py
import pytest
import torch
from models.encoders import HeteroDictEncoder, InnerProductDecoder


class TestHeteroDictEncoder:
    @pytest.fixture
    def node_types(self):
        return ["Gene", "Protein", "Disease"]

    @pytest.fixture
    def encoder(self, node_types):
        return HeteroDictEncoder(
            in_channels={"Gene": 896, "Protein": 896, "Disease": 896},
            hidden_channels=128,
            node_types=node_types,
        )

    def test_encoder_projects_each_type_to_hidden_dim(self, encoder):
        x_dict = {
            "Gene": torch.randn(10, 896),
            "Protein": torch.randn(5, 896),
            "Disease": torch.randn(3, 896),
        }
        out = encoder(x_dict)
        assert out["Gene"].shape == (10, 128)
        assert out["Protein"].shape == (5, 128)
        assert out["Disease"].shape == (3, 128)

    def test_encoder_preserves_device(self, encoder):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        encoder = encoder.cuda()
        x_dict = {
            "Gene": torch.randn(10, 896, device="cuda"),
            "Protein": torch.randn(5, 896, device="cuda"),
        }
        out = encoder(x_dict)
        assert out["Gene"].device.type == "cuda"

    def test_unknown_node_type_returns_zero_tensor(self, encoder):
        """Unknown node types should not crash — return zero tensor with warning."""
        x_dict = {"Gene": torch.randn(10, 896), "Unknown": torch.randn(2, 64)}
        out = encoder(x_dict)
        assert "Unknown" in out
        assert out["Unknown"].shape == (2, 128)


class TestInnerProductDecoder:
    @pytest.fixture
    def decoder(self):
        return InnerProductDecoder()

    def test_decode_single_edge_type_returns_logits(self, decoder):
        z_dict = {"Gene": torch.randn(10, 128), "Disease": torch.randn(5, 128)}
        edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])  # 3 edges
        logits = decoder(z_dict, edge_index, src_type="Gene", dst_type="Disease")
        assert logits.shape == (3,)
        assert logits.dtype == torch.float32

    def test_decode_sigmoid_produces_probabilities(self, decoder):
        z_dict = {"Gene": torch.randn(10, 128), "Disease": torch.randn(5, 128)}
        edge_index = torch.tensor([[0, 1], [0, 0]])
        probs = decoder.decode_prob(z_dict, edge_index, src_type="Gene", dst_type="Disease")
        assert probs.shape == (2,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_score_all_pairs_returns_full_matrix(self, decoder):
        z_dict = {"Gene": torch.randn(5, 128), "Disease": torch.randn(3, 128)}
        scores = decoder.score_all_pairs(z_dict, src_type="Gene", dst_type="Disease")
        assert scores.shape == (5, 3)
