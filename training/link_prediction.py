# training/link_prediction.py
"""Link Prediction training loop for Tempered HGT.

CRITICAL: Message-passing graph (train edges only) is strictly separated
from evaluation edges (val/test). Val/test edges are NEVER used for
neighborhood aggregation — they only receive scores after node embeddings
are computed from the train graph.
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

    Architecture:
      - train_epoch(): MP on train_ei, decode train_ei vs train_neg_ei
      - evaluate():   MP on train_ei, decode eval_ei vs eval_neg_ei
      - test():       MP on train+val_ei, decode test_ei vs test_neg_ei

    This ensures val/test edges are NEVER seen during message passing.
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
        self.default_feat_dim = 896

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.cos_decay = CosineAnnealingDecay(total_steps=num_epochs)

        self.best_val_mrr = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def _build_x_dict(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Build feature dict for node types known to the model."""
        x_dict = {}
        for nt in data.node_types:
            # Only include node types the model knows about (in its encoder)
            if hasattr(self.model, 'encoder') and nt not in self.model.encoder.projections:
                continue
            if hasattr(data[nt], 'x') and data[nt].x is not None and data[nt].num_nodes > 0:
                x_dict[nt] = data[nt].x
        return x_dict

    def _bce_loss(self, pos_logits: torch.Tensor,
                  neg_logits: torch.Tensor) -> torch.Tensor:
        """BCE loss: positive edges -> 1, negative edges -> 0."""
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([pos_labels, neg_labels])
        return self.loss_fn(logits, labels)

    def train_epoch(self, data: HeteroData, train_ei: Dict,
                    train_neg_ei: Dict, epoch: int) -> dict:
        """Train one epoch.

        Message passing on train_ei ONLY. Decode train edges vs negatives.
        """
        self.model.train()
        cos_decay = self.cos_decay(epoch)
        x_dict = self._build_x_dict(data)

        z_dict = self.model(
            x_dict, train_ei,
            cos_decay=cos_decay,
            edge_weight_bias=self.edge_weight_bias,
        )

        # Collect all logits across edge types, compute BCE once.
        # Avoids multi-loss backward issues when losses share z_dict graph.
        all_pos_logits = []
        all_neg_logits = []
        total_loss = 0.0
        n_batches = 0

        for edge_type in train_ei:
            src_t, rel, dst_t = edge_type
            if src_t not in z_dict or dst_t not in z_dict:
                continue
            pos_ei = train_ei[edge_type]
            neg_ei_typed = train_neg_ei.get(edge_type)
            if neg_ei_typed is None:
                continue

            pos_logits = self.model.decode(z_dict, pos_ei, src_t, dst_t)
            neg_logits = self.model.decode(z_dict, neg_ei_typed, src_t, dst_t)
            all_pos_logits.append(pos_logits)
            all_neg_logits.append(neg_logits)
            n_batches += 1

        if n_batches > 0:
            pos = torch.cat(all_pos_logits)
            neg = torch.cat(all_neg_logits)
            loss = self._bce_loss(pos, neg)
            total_loss = loss.item()

            self.optimizer.zero_grad()
            params = list(self.model.parameters())
            grads = torch.autograd.grad(loss, params, retain_graph=False,
                                        allow_unused=True)
            for p, g in zip(params, grads):
                if g is not None:
                    p.grad = g
            self.optimizer.step()

        return {"loss": total_loss}

    @torch.no_grad()
    def evaluate(self, data: HeteroData, msg_ei: Dict,
                 eval_ei: Dict, eval_neg_ei: Dict) -> dict:
        """Evaluate on held-out edges WITHOUT using them for message passing.

        Args:
            data: HeteroData with node features
            msg_ei: Message-passing edges (TRAIN only — NEVER val/test)
            eval_ei: Edges to score (val or test)
            eval_neg_ei: Negative edges for eval scoring
        """
        self.model.eval()
        x_dict = self._build_x_dict(data)

        # Message passing uses ONLY train edges
        z_dict = self.model(x_dict, msg_ei, cos_decay=0.0)

        all_pos_logits = []
        all_neg_logits = []

        for edge_type in eval_ei:
            src_t, rel, dst_t = edge_type
            pos_ei = eval_ei[edge_type]
            neg_ei_typed = eval_neg_ei.get(edge_type)
            if neg_ei_typed is None:
                continue

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

    @torch.no_grad()
    def test(self, data: HeteroData, train_ei: Dict, val_ei: Dict,
             test_ei: Dict, test_neg_ei: Dict) -> dict:
        """Final test evaluation.

        Standard transductive setting: message passing on train+val edges
        to predict test edges. Both endpoints of test edges must exist in
        the combined train+val graph.
        """
        mp_graph = {}
        all_edge_types = set(train_ei.keys()) | set(val_ei.keys())
        for et in all_edge_types:
            train_part = train_ei.get(et, torch.empty((2, 0), dtype=torch.long))
            val_part = val_ei.get(et, torch.empty((2, 0), dtype=torch.long))
            mp_graph[et] = torch.cat([train_part, val_part], dim=1)
        return self.evaluate(data, mp_graph, test_ei, test_neg_ei)

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

    def fit(self, data: HeteroData,
            train_ei: Dict, val_ei: Dict, test_ei: Dict = None,
            train_neg_ei: Dict = None, val_neg_ei: Dict = None,
            test_neg_ei: Dict = None,
            verbose: bool = True) -> dict:
        """Full training loop with proper train/val/test separation.

        Args:
            data: HeteroData with node features attached
            train_ei: Training edges (used for message passing AND scoring)
            val_ei: Validation edges (scored only, NEVER in message passing)
            test_ei: Test edges (scored only at final evaluation)
            train_neg_ei: Negative edges paired with training edges
            val_neg_ei: Negative edges paired with validation edges
            test_neg_ei: Negative edges paired with test edges

        Returns:
            {"best_epoch": int, "best_val_mrr": float, "history": list,
             "test_metrics": dict (if test_ei provided)}
        """
        history = []

        for epoch in range(self.num_epochs):
            train_metrics = self.train_epoch(data, train_ei, train_neg_ei, epoch)
            val_metrics = self.evaluate(data, train_ei, val_ei, val_neg_ei)

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

        result = {
            "best_epoch": self.best_epoch,
            "best_val_mrr": self.best_val_mrr,
            "history": history,
        }

        # Final test evaluation
        if test_ei is not None and test_neg_ei is not None:
            # Load best checkpoint for test eval
            best_ckpt = self.checkpoint_dir / f"checkpoint_epoch_{self.best_epoch}.pt"
            if best_ckpt.exists():
                ckpt = torch.load(best_ckpt, weights_only=True, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
            test_metrics = self.test(data, train_ei, val_ei, test_ei, test_neg_ei)
            result["test_metrics"] = test_metrics
            if verbose:
                print(f"Test AUROC: {test_metrics['auroc']:.4f} | "
                      f"Test MRR: {test_metrics['mrr']:.4f} | "
                      f"Test Hits@10: {test_metrics['hits@10']:.4f}")

        return result
