# training/link_prediction.py
"""Mini-batch Link Prediction training loop for Tempered HGT.

Supports: BCE loss, Cosine Annealing decay of edge bias, Early stopping,
Checkpointing. Mini-batch via LinkNeighborLoader (used in full training).
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
from torch_geometric.data import HeteroData

from .edge_bias import CosineAnnealingDecay
from .metrics import compute_auroc, compute_mrr, compute_hits_at_k


class LinkPredictionTrainer:
    """Training loop for heterogeneous graph link prediction.

    Args:
        model: TemperedHGT model
        learning_rate: AdamW learning rate
        num_epochs: Maximum training epochs
        patience: Early stopping patience (epochs without val MRR improvement)
        device: "cpu" or "cuda"
        checkpoint_dir: Directory for model checkpoints
        edge_weight_bias: Pre-computed edge bias dict from EdgeBiasInitializer
        batch_size: Mini-batch size (default 256)
        num_neighbors: Neighbor sampling fanout (default [10, 5, 5])
    """

    def __init__(self, model: nn.Module, learning_rate: float = 1e-3,
                 num_epochs: int = 100, patience: int = 10,
                 device: str = "cpu", checkpoint_dir: str = "checkpoints",
                 edge_weight_bias: Optional[Dict] = None,
                 batch_size: int = 256,
                 num_neighbors: list = None):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.edge_weight_bias = edge_weight_bias or {}
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors or [10, 5, 5]

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.cos_decay = CosineAnnealingDecay(total_steps=num_epochs)

        # Early stopping state
        self.best_val_mrr = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def _bce_loss(self, pos_logits: torch.Tensor,
                  neg_logits: torch.Tensor) -> torch.Tensor:
        """BCE loss: positive edges -> 1, negative edges -> 0."""
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([pos_labels, neg_labels])
        return self.loss_fn(logits, labels)

    def train_epoch(self, data: HeteroData, train_ei: Dict,
                    neg_ei: Dict, epoch: int) -> dict:
        """Train one epoch. Returns metrics dict."""
        self.model.train()
        cos_decay = self.cos_decay(epoch)

        # Build feature dict once
        x_dict = {}
        for nt in data.node_types:
            if hasattr(data[nt], 'x') and data[nt].x is not None:
                x_dict[nt] = data[nt].x
            else:
                x_dict[nt] = torch.randn(data[nt].num_nodes, 896)

        # Single forward pass with ALL edge types
        z_dict = self.model(
            x_dict, train_ei,
            cos_decay=cos_decay,
            edge_weight_bias=self.edge_weight_bias,
        )

        # Compute combined loss across all edge types, backward once
        losses = []
        total_loss = 0.0
        n_batches = 0

        for edge_type in train_ei:
            src_t, rel, dst_t = edge_type
            pos_ei = train_ei[edge_type]
            neg_ei_typed = neg_ei.get(edge_type)
            if neg_ei_typed is None:
                continue

            pos_logits = self.model.decode(z_dict, pos_ei, src_t, dst_t)
            neg_logits = self.model.decode(z_dict, neg_ei_typed, src_t, dst_t)

            loss = self._bce_loss(pos_logits, neg_logits)
            losses.append(loss)
            total_loss += loss.item()
            n_batches += 1

        if losses:
            combined = torch.stack(losses).sum()
            self.optimizer.zero_grad()
            combined.backward()
            self.optimizer.step()

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate(self, data: HeteroData, val_ei: Dict,
                 neg_ei: Dict) -> dict:
        """Evaluate on validation set."""
        self.model.eval()

        all_pos_logits = []
        all_neg_logits = []

        for edge_type in val_ei:
            src_t, rel, dst_t = edge_type
            pos_ei = val_ei[edge_type]
            neg_ei_typed = neg_ei.get(edge_type)
            if neg_ei_typed is None:
                continue

            x_dict = {}
            for nt in data.node_types:
                if hasattr(data[nt], 'x') and data[nt].x is not None:
                    x_dict[nt] = data[nt].x
                else:
                    x_dict[nt] = torch.randn(data[nt].num_nodes, 896)

            z_dict = self.model(
                x_dict, {edge_type: pos_ei},
                cos_decay=0.0,  # No bias at evaluation
            )

            pos_logits = self.model.decode(z_dict, pos_ei, src_t, dst_t)
            neg_logits = self.model.decode(z_dict, neg_ei_typed, src_t, dst_t)

            all_pos_logits.append(pos_logits)
            all_neg_logits.append(neg_logits)

        if not all_pos_logits:
            return {"auroc": 0.5, "mrr": 0.0, "hits@10": 0.0}

        pos = torch.cat(all_pos_logits)
        neg = torch.cat(all_neg_logits)

        return {
            "auroc": compute_auroc(pos, neg),
            "mrr": compute_mrr(pos, neg),
            "hits@10": compute_hits_at_k(pos.unsqueeze(1), neg.repeat(pos.shape[0], 1), k=10),
        }

    def _should_stop_early(self) -> bool:
        return self.patience_counter >= self.patience

    def _save_checkpoint(self, epoch: int, val_mrr: float):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_mrr": val_mrr,
            "best_val_mrr": self.best_val_mrr,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(ckpt, path)

    def fit(self, data: HeteroData, train_ei: Dict, val_ei: Dict,
            neg_ei: Dict, verbose: bool = True) -> dict:
        """Full training loop.

        Returns: {"best_epoch": int, "best_val_mrr": float, "history": list}
        """
        history = []

        for epoch in range(self.num_epochs):
            train_metrics = self.train_epoch(data, train_ei, neg_ei, epoch)
            val_metrics = self.evaluate(data, val_ei, neg_ei)

            metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            metrics["epoch"] = epoch
            history.append(metrics)

            if val_metrics["mrr"] > self.best_val_mrr:
                self.best_val_mrr = val_metrics["mrr"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics["mrr"])
            else:
                self.patience_counter += 1

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | loss: {train_metrics['loss']:.4f} | "
                      f"val_auroc: {val_metrics['auroc']:.3f} | "
                      f"val_mrr: {val_metrics['mrr']:.3f}")

            if self._should_stop_early():
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        return {
            "best_epoch": self.best_epoch,
            "best_val_mrr": self.best_val_mrr,
            "history": history,
        }
