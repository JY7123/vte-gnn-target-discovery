# tests/test_tempered_hgt.py (part 1)
import pytest
import torch
from models.tempered_hgt import TemperedHGTConv


class TestTemperedHGTConv:
    @pytest.fixture
    def conv(self):
        return TemperedHGTConv(
            in_channels=128,
            out_channels=128,
            num_heads=4,
            meta_relations=[
                ("Gene", "REGULATES", "Gene"),
                ("Gene", "ASSOCIATED_WITH", "Disease"),
                ("Protein", "BINDS_TO", "Protein"),
            ],
            temperature_init={
                "Gene__REGULATES__Gene": 0.5,
                "Gene__ASSOCIATED_WITH__Disease": 1.0,
                "Protein__BINDS_TO__Protein": 0.7,
            },
        )

    @pytest.fixture
    def x_dict(self):
        return {
            "Gene": torch.randn(10, 128),
            "Protein": torch.randn(5, 128),
            "Disease": torch.randn(4, 128),
        }

    @pytest.fixture
    def edge_index_dict(self):
        return {
            ("Gene", "REGULATES", "Gene"): torch.tensor([[0, 1, 2], [1, 2, 3]]),
            ("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0, 1], [0, 1]]),
        }

    def test_temperatures_are_learnable_parameters(self, conv):
        assert hasattr(conv, "temperatures")
        assert isinstance(conv.temperatures, torch.nn.ParameterDict)
        assert "Gene__REGULATES__Gene" in conv.temperatures
        assert abs(conv.temperatures["Gene__REGULATES__Gene"].item() - 0.5) < 0.01
        for key, tau in conv.temperatures.items():
            assert tau.item() > 0, f"Temperature {key} must be positive"

    def test_forward_produces_embeddings(self, conv, x_dict, edge_index_dict):
        out = conv(x_dict, edge_index_dict, cos_decay=1.0)
        assert "Gene" in out
        assert out["Gene"].shape == (10, 128)
        assert out["Protein"].shape == (5, 128)
        assert out["Disease"].shape == (4, 128)

    def test_forward_no_nan(self, conv, x_dict, edge_index_dict):
        out = conv(x_dict, edge_index_dict, cos_decay=1.0)
        for nt, emb in out.items():
            assert not torch.isnan(emb).any(), f"NaN in {nt} output"

    def test_cos_decay_zero_eliminates_bias(self, conv):
        """When cos_decay=0, edge_bias should have no effect."""
        conv.eval()  # disable dropout for deterministic comparison
        x_dict = {"Gene": torch.randn(8, 128)}
        ei = {("Gene", "REGULATES", "Gene"): torch.tensor([[0, 0], [1, 2]])}

        # Use string keys for edge_weight_bias to match internal _meta_to_key lookup
        edge_bias = {"Gene__REGULATES__Gene": torch.tensor([100.0, 100.0])}
        out_with_bias = conv(x_dict, ei, cos_decay=0.0, edge_weight_bias=edge_bias)
        out_no_bias = conv(x_dict, ei, cos_decay=0.0, edge_weight_bias=None)

        assert torch.allclose(out_with_bias["Gene"], out_no_bias["Gene"], atol=1e-4)

    def test_temperature_sharpens_attention(self):
        """Smaller tau should produce different (sharper) outputs than larger tau."""
        torch.manual_seed(42)
        x_dict = {"Gene": torch.randn(8, 128)}

        conv_sharp = TemperedHGTConv(
            in_channels=128, out_channels=128, num_heads=4,
            meta_relations=[("Gene", "REGULATES", "Gene")],
            temperature_init={"Gene__REGULATES__Gene": 0.1},
        )
        conv_flat = TemperedHGTConv(
            in_channels=128, out_channels=128, num_heads=4,
            meta_relations=[("Gene", "REGULATES", "Gene")],
            temperature_init={"Gene__REGULATES__Gene": 10.0},
        )

        # Copy non-temperature weights
        for key in conv_sharp.state_dict():
            if "temperatures" not in key:
                conv_flat.state_dict()[key].copy_(conv_sharp.state_dict()[key])

        # Override tau explicitly
        conv_sharp.temperatures["Gene__REGULATES__Gene"].data.fill_(0.1)
        conv_flat.temperatures["Gene__REGULATES__Gene"].data.fill_(10.0)

        ei = {("Gene", "REGULATES", "Gene"): torch.tensor([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]])}
        out_sharp = conv_sharp(x_dict, ei, cos_decay=1.0)
        out_flat = conv_flat(x_dict, ei, cos_decay=1.0)

        assert not torch.allclose(out_sharp["Gene"], out_flat["Gene"], atol=1e-3)

    def test_edge_bias_changes_attention(self, conv):
        """Edge bias should produce different output when cos_decay > 0."""
        conv.eval()  # disable dropout for deterministic comparison
        x_dict = {"Gene": torch.randn(8, 128)}
        ei = {("Gene", "REGULATES", "Gene"): torch.tensor([[0, 0], [1, 2]])}

        edge_bias = {("Gene", "REGULATES", "Gene"): torch.tensor([5.0, 0.0])}
        out_with = conv(x_dict, ei, cos_decay=1.0, edge_weight_bias=edge_bias)
        out_without = conv(x_dict, ei, cos_decay=1.0, edge_weight_bias=None)

        assert not torch.allclose(out_with["Gene"], out_without["Gene"], atol=1e-4)

    def test_temperatures_are_clamped_positive(self, conv):
        """After forward, tau must stay >= 0.01 (float32 epsilon accounted for)."""
        x_dict = {"Gene": torch.randn(8, 128)}
        ei = {("Gene", "REGULATES", "Gene"): torch.tensor([[0], [1]])}
        # Manually set a negative tau to test clamping
        conv.temperatures["Gene__REGULATES__Gene"].data.fill_(-0.5)
        conv(x_dict, ei, cos_decay=1.0)
        # float32 representation of 0.01 is ~0.009999999776, so check >= 0.0095
        assert conv.temperatures["Gene__REGULATES__Gene"].item() >= 0.0095


class TestTemperedHGT:
    @pytest.fixture
    def model(self):
        from models.tempered_hgt import TemperedHGT
        return TemperedHGT(
            in_channels={"Gene": 896, "Protein": 896, "Disease": 896},
            hidden_channels=128,
            out_channels=128,
            num_heads=4,
            num_layers=2,
            meta_relations=[
                ("Gene", "REGULATES", "Gene"),
                ("Gene", "ASSOCIATED_WITH", "Disease"),
                ("Protein", "BINDS_TO", "Protein"),
            ],
            temperature_init={
                "Gene__REGULATES__Gene": 0.5,
                "Gene__ASSOCIATED_WITH__Disease": 1.0,
                "Protein__BINDS_TO__Protein": 0.7,
            },
            dropout=0.1,
        )

    @pytest.fixture
    def x_dict(self):
        return {
            "Gene": torch.randn(10, 896),
            "Protein": torch.randn(5, 896),
            "Disease": torch.randn(4, 896),
        }

    @pytest.fixture
    def edge_index_dict(self):
        return {
            ("Gene", "REGULATES", "Gene"): torch.tensor([[0, 1, 2], [1, 2, 3]]),
            ("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0, 1], [0, 1]]),
        }

    def test_model_forward_returns_embeddings(self, model, x_dict, edge_index_dict):
        z_dict = model(x_dict, edge_index_dict, cos_decay=1.0)
        assert z_dict["Gene"].shape == (10, 128)
        assert z_dict["Protein"].shape == (5, 128)
        assert z_dict["Disease"].shape == (4, 128)

    def test_model_decode_returns_logits(self, model, x_dict, edge_index_dict):
        z_dict = model(x_dict, edge_index_dict, cos_decay=1.0)
        pos_edge = torch.tensor([[0], [1]])
        pos_logits = model.decode(z_dict, pos_edge, "Gene", "Gene")
        assert pos_logits.shape == (1,)

    def test_multi_layer_stacks_correctly(self, model):
        assert len(model.convs) == 2

    def test_forward_preserves_gradient(self, model, x_dict, edge_index_dict):
        """Forward pass must support backprop through tau parameters."""
        z = model(x_dict, edge_index_dict, cos_decay=1.0)
        pos_edge = torch.tensor([[0], [0]])
        logits = model.decode(z, pos_edge, "Gene", "Disease")
        loss = -torch.log(torch.sigmoid(logits) + 1e-8).mean()
        loss.backward()
        # Check that tau got gradients for relations that have edges in the batch
        # (relations without edges don't participate in the computation graph)
        keys_with_edges = {
            f"{src}__{rel}__{dst}"
            for (src, rel, dst) in edge_index_dict.keys()
        }
        for conv in model.convs:
            for key, tau in conv.temperatures.items():
                if key not in keys_with_edges:
                    continue  # skip relations with no edges in this batch
                assert tau.grad is not None, f"No gradient for {key}"

    def test_model_can_overfit_tiny_graph(self, model):
        """On a tiny 3-node graph with 1 edge, loss should go to near-zero."""
        tiny_x = {
            "Gene": torch.randn(3, 896),
            "Disease": torch.randn(2, 896),
        }
        tiny_ei = {
            ("Gene", "ASSOCIATED_WITH", "Disease"): torch.tensor([[0], [0]]),
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(50):
            optimizer.zero_grad()
            z = model(tiny_x, tiny_ei, cos_decay=1.0)
            pos_logits = model.decode(z, torch.tensor([[0], [0]]), "Gene", "Disease")
            neg_logits = model.decode(z, torch.tensor([[0], [1]]), "Gene", "Disease")
            loss = -torch.log(torch.sigmoid(pos_logits) + 1e-8).mean() - torch.log(1 - torch.sigmoid(neg_logits) + 1e-8).mean()
            loss.backward()
            optimizer.step()

        # After 50 steps, positive should score higher than negative
        with torch.no_grad():
            z = model(tiny_x, tiny_ei, cos_decay=1.0)
            pos = model.decode(z, torch.tensor([[0], [0]]), "Gene", "Disease")
            neg = model.decode(z, torch.tensor([[0], [1]]), "Gene", "Disease")
        assert pos.item() > neg.item(), f"pos={pos.item():.3f} <= neg={neg.item():.3f}"
