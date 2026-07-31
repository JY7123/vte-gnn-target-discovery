# training/baselines.py
"""KG embedding baselines for link prediction benchmarking.

Implements TransE, DistMult, ComplEx, RotatE in pure PyTorch.
Each baseline operates on the same train/val/test split and reports
filtered MRR/Hits@K for fair comparison with TemperedHGT.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from .metrics import filtered_evaluation, build_true_triples_set


class TransE(nn.Module):
    """Bordes et al. 2013: h + r ≈ t, score = -||h + r - t||."""

    def __init__(self, num_entities: int, num_relations: int, dim: int = 128,
                 margin: float = 1.0):
        super().__init__()
        self.dim = dim
        self.margin = margin
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        # Normalize entity embeddings
        with torch.no_grad():
            self.entity_emb.weight.data = nn.functional.normalize(
                self.entity_emb.weight.data, p=2, dim=1
            )

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.rel_emb(r_idx)
        t = self.entity_emb(t_idx)
        return -torch.norm(h + r - t, p=2, dim=1)

    def score_all_tails(self, h_idx, r_idx, num_entities: int):
        h = self.entity_emb(h_idx)  # [B, dim]
        r = self.rel_emb(r_idx)     # [B, dim]
        all_ent = self.entity_emb.weight  # [N, dim]
        # For each (h,r), compute -||h+r - t|| for all t
        hr = h + r  # [B, dim]
        # Expand and compute distances: [B, N]
        hr_exp = hr.unsqueeze(1)        # [B, 1, dim]
        t_exp = all_ent.unsqueeze(0)     # [1, N, dim]
        return -torch.norm(hr_exp - t_exp, p=2, dim=2)

    def score_all_heads(self, t_idx, r_idx, num_entities: int):
        t = self.entity_emb(t_idx)
        r = self.rel_emb(r_idx)
        all_ent = self.entity_emb.weight
        t_minus_r = t - r
        t_exp = t_minus_r.unsqueeze(1)
        h_exp = all_ent.unsqueeze(0)
        return -torch.norm(h_exp - t_exp, p=2, dim=2)


class DistMult(nn.Module):
    """Yang et al. 2015: score = h^T diag(r) t, -symmetric."""

    def __init__(self, num_entities: int, num_relations: int, dim: int = 128):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.rel_emb(r_idx)
        t = self.entity_emb(t_idx)
        return (h * r * t).sum(dim=1)

    def score_all_tails(self, h_idx, r_idx, num_entities: int):
        h = self.entity_emb(h_idx)    # [B, dim]
        r = self.rel_emb(r_idx)       # [B, dim]
        all_ent = self.entity_emb.weight  # [N, dim]
        hr = h * r                     # [B, dim]
        return hr @ all_ent.T         # [B, N]

    def score_all_heads(self, t_idx, r_idx, num_entities: int):
        t = self.entity_emb(t_idx)
        r = self.rel_emb(r_idx)
        all_ent = self.entity_emb.weight
        tr = t * r
        return tr @ all_ent.T


class ComplEx(nn.Module):
    """Trouillon et al. 2016: complex embeddings, Re(h * r * conj(t))."""

    def __init__(self, num_entities: int, num_relations: int, dim: int = 128):
        super().__init__()
        self.dim = dim
        # Store as [real, imag] concatenated: dim = 2 * half_dim
        self.half_dim = dim // 2
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def _re_im(self, x):
        return x[..., :self.half_dim], x[..., self.half_dim:]

    def forward(self, h_idx, r_idx, t_idx):
        h_re, h_im = self._re_im(self.entity_emb(h_idx))
        r_re, r_im = self._re_im(self.rel_emb(r_idx))
        t_re, t_im = self._re_im(self.entity_emb(t_idx))
        # Re(<h, r, conj(t)>) = <h_re, r_re, t_re> + <h_re, r_im, t_im>
        #                         + <h_im, r_re, t_im> - <h_im, r_im, t_re>
        return ((h_re * r_re * t_re).sum(dim=1) +
                (h_re * r_im * t_im).sum(dim=1) +
                (h_im * r_re * t_im).sum(dim=1) -
                (h_im * r_im * t_re).sum(dim=1))

    def score_all_tails(self, h_idx, r_idx, num_entities: int):
        h = self.entity_emb(h_idx)
        r = self.rel_emb(r_idx)
        h_re, h_im = self._re_im(h)
        r_re, r_im = self._re_im(r)
        all_ent = self.entity_emb.weight
        t_re, t_im = self._re_im(all_ent)
        return (torch.mm(h_re * r_re, t_re.T) +
                torch.mm(h_re * r_im, t_im.T) +
                torch.mm(h_im * r_re, t_im.T) -
                torch.mm(h_im * r_im, t_re.T))

    def score_all_heads(self, t_idx, r_idx, num_entities: int):
        t = self.entity_emb(t_idx)
        r = self.rel_emb(r_idx)
        t_re, t_im = self._re_im(t)
        r_re, r_im = self._re_im(r)
        all_ent = self.entity_emb.weight
        h_re, h_im = self._re_im(all_ent)
        return (torch.mm(t_re * r_re, h_re.T) +
                torch.mm(t_re * r_im, h_im.T) +
                torch.mm(t_im * r_re, h_im.T) -
                torch.mm(t_im * r_im, h_re.T))


class RotatE(nn.Module):
    """Sun et al. 2019: rotation in complex space, h ∘ r = t."""

    def __init__(self, num_entities: int, num_relations: int, dim: int = 128,
                 margin: float = 6.0):
        super().__init__()
        self.dim = dim
        self.margin = margin
        self.half_dim = dim // 2
        # Entity: [re, im] concatenated
        self.entity_emb = nn.Embedding(num_entities, dim)
        # Relation: phase angles only (dim // 2 values)
        self.rel_emb = nn.Embedding(num_relations, self.half_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.uniform_(self.rel_emb.weight, -3.14159, 3.14159)

    def _re_im(self, x):
        return x[..., :self.half_dim], x[..., self.half_dim:]

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r_phase = self.rel_emb(r_idx)
        t = self.entity_emb(t_idx)
        h_re, h_im = self._re_im(h)
        t_re, t_im = self._re_im(t)
        r_re, r_im = torch.cos(r_phase), torch.sin(r_phase)
        # Rotation: h_rot = (h_re*r_re - h_im*r_im, h_re*r_im + h_im*r_re)
        h_rot_re = h_re * r_re - h_im * r_im
        h_rot_im = h_re * r_im + h_im * r_re
        return -torch.norm(
            torch.cat([h_rot_re - t_re, h_rot_im - t_im], dim=1), p=2, dim=1
        )

    def score_all_tails(self, h_idx, r_idx, num_entities: int):
        """Score all tails using expansion: ||h_rot - t||^2 = ||h_rot||^2 + ||t||^2 - 2*<h_rot, t>.

        Avoids [B, N, D] intermediate tensors (OOM with 82k entities).
        """
        h = self.entity_emb(h_idx)
        r_phase = self.rel_emb(r_idx)
        h_re, h_im = self._re_im(h)
        r_re, r_im = torch.cos(r_phase), torch.sin(r_phase)
        h_rot_re = h_re * r_re - h_im * r_im      # [B, D/2]
        h_rot_im = h_re * r_im + h_im * r_re      # [B, D/2]
        all_ent = self.entity_emb.weight           # [N, D]
        t_re, t_im = self._re_im(all_ent)          # [N, D/2]

        # ||h_rot||^2 = sum(h_rot_re^2 + h_rot_im^2) per row → [B]
        h_norm2 = (h_rot_re ** 2).sum(dim=1) + (h_rot_im ** 2).sum(dim=1)
        # ||t||^2 = sum(t_re^2 + t_im^2) per row → [N]
        t_norm2 = (t_re ** 2).sum(dim=1) + (t_im ** 2).sum(dim=1)
        # <h_rot, t> = h_rot_re @ t_re.T + h_rot_im @ t_im.T → [B, N]
        dot = h_rot_re @ t_re.T + h_rot_im @ t_im.T

        dist2 = h_norm2.unsqueeze(1) + t_norm2.unsqueeze(0) - 2 * dot
        dist2.clamp_(min=0)  # avoid negative due to numerical error
        return -torch.sqrt(dist2)

    def score_all_heads(self, t_idx, r_idx, num_entities: int):
        """Score all heads using expansion (inverse rotation).

        Returns [B, N] where B = len(t_idx), N = num_entities.
        """
        t = self.entity_emb(t_idx)
        r_phase = self.rel_emb(r_idx)
        t_re, t_im = self._re_im(t)
        r_re, r_im = torch.cos(r_phase), torch.sin(r_phase)
        # Inverse rotation: expected head h = rotate(t, -r)
        h_exp_re = t_re * r_re + t_im * r_im   # [B, D/2]
        h_exp_im = -t_re * r_im + t_im * r_re  # [B, D/2]
        all_ent = self.entity_emb.weight
        h_re, h_im = self._re_im(all_ent)      # [N, D/2]

        h_exp_norm2 = (h_exp_re ** 2).sum(dim=1) + (h_exp_im ** 2).sum(dim=1)  # [B]
        h_norm2 = (h_re ** 2).sum(dim=1) + (h_im ** 2).sum(dim=1)              # [N]
        # h_exp @ h.T → [B, N]
        dot = h_exp_re @ h_re.T + h_exp_im @ h_im.T

        dist2 = h_exp_norm2.unsqueeze(1) + h_norm2.unsqueeze(0) - 2 * dot  # [B, N]
        dist2.clamp_(min=0)
        return -torch.sqrt(dist2)


class BaselineTrainer:
    """Unified training loop for KG embedding baselines.

    Maps heterogeneous graph edge types to integer relation indices.
    Uses margin ranking loss with negative sampling.
    """

    def __init__(self, model: nn.Module, learning_rate: float = 1e-3,
                 num_epochs: int = 100, device: str = "cpu",
                 num_negatives: int = 1):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.num_negatives = num_negatives
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _sample_negatives(self, triples: torch.Tensor, num_entities: int):
        """Sample negative triples by corrupting tail."""
        n = triples.shape[0]
        neg_tails = torch.randint(0, num_entities, (n * self.num_negatives,),
                                   device=triples.device)
        neg = triples.repeat(self.num_negatives, 1).clone()
        neg[:, 2] = neg_tails
        return neg

    def train_epoch(self, triples: torch.Tensor, num_entities: int) -> float:
        self.model.train()
        neg = self._sample_negatives(triples, num_entities)
        pos_scores = self.model(triples[:, 0], triples[:, 1], triples[:, 2])
        neg_scores = self.model(neg[:, 0], neg[:, 1], neg[:, 2])

        # Margin ranking loss for TransE/RotatE, BCE for DistMult/ComplEx
        if isinstance(self.model, (TransE, RotatE)):
            margin = self.model.margin
            target = torch.ones_like(pos_scores)
            loss = nn.functional.margin_ranking_loss(
                pos_scores, neg_scores, target, margin=margin
            )
        else:
            labels = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores),
            ])
            logits = torch.cat([pos_scores, neg_scores])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, triples: torch.Tensor, num_entities: int,
            verbose: bool = True) -> list:
        history = []
        for epoch in range(self.num_epochs):
            loss = self.train_epoch(triples, num_entities)
            history.append({"epoch": epoch, "loss": loss})
            if verbose and epoch % 20 == 0:
                print(f"  Epoch {epoch:3d} | loss: {loss:.4f}")
        return history


def build_triple_mapping(
    train_ei: Dict, val_ei: Dict, test_ei: Dict, data
) -> Tuple[Dict, Dict, int, List]:
    """Map heterogeneous edge types to global entity IDs and relation IDs.

    Returns:
        global_id_map: {(node_type, local_idx) -> global_idx}
        rel_map: {edge_type_tuple -> rel_int_id}
        num_entities: int (total across all types)
        all_triples: [(h, r, t), ...] for training
    """
    global_id_map = {}
    rel_map = {}
    current_id = 0

    for nt in data.node_types:
        for local_idx in range(data[nt].num_nodes):
            global_id_map[(nt, local_idx)] = current_id
            current_id += 1

    num_entities = current_id

    # Assign relation IDs
    rel_list = sorted(set(list(train_ei.keys()) + list(val_ei.keys()) + list(test_ei.keys())))
    for i, et in enumerate(rel_list):
        rel_map[et] = i

    return global_id_map, rel_map, num_entities


def edges_to_triples(ei_dict: Dict, rel_map: Dict, global_id_map: Dict) -> torch.Tensor:
    """Convert edge dict to triple tensor [N, 3] (h, r, t)."""
    triples = []
    for et, ei in ei_dict.items():
        if et not in rel_map:
            continue
        src_t, rel_str, dst_t = et
        r = rel_map[et]
        for j in range(ei.shape[1]):
            h = global_id_map.get((src_t, int(ei[0, j])))
            t = global_id_map.get((dst_t, int(ei[1, j])))
            if h is not None and t is not None:
                triples.append((h, r, t))
    if not triples:
        return torch.empty((0, 3), dtype=torch.long)
    return torch.tensor(triples, dtype=torch.long)


def evaluate_baseline_filtered(
    model, test_triples: torch.Tensor, all_true_triples: Set[Tuple[int, int, int]],
    num_entities: int, batch_size: int = 256
) -> dict:
    """Compute filtered MRR/Hits@K for a baseline model.

    Uses the model's score_all_tails / score_all_heads methods.
    """
    model.eval()
    n = test_triples.shape[0]
    tail_ranks = []
    head_ranks = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = test_triples[start:end]
            h_idx = batch[:, 0]
            r_idx = batch[:, 1]
            t_idx = batch[:, 2]

            # Tail prediction
            tail_scores = model.score_all_tails(h_idx, r_idx, num_entities)  # [B, N]
            for i in range(end - start):
                scores = tail_scores[i].clone()
                # Filter known true triples for this (h, r)
                for hh, rr, tt in all_true_triples:
                    if hh == h_idx[i].item() and rr == r_idx[i].item() and tt != t_idx[i].item():
                        if tt < num_entities:
                            scores[tt] = float('-inf')
                true_score = scores[t_idx[i].item()]
                rank = (scores > true_score).sum().item() + 1
                tail_ranks.append(rank)

            # Head prediction
            head_scores = model.score_all_heads(t_idx, r_idx, num_entities)
            for i in range(end - start):
                scores = head_scores[i].clone()
                for hh, rr, tt in all_true_triples:
                    if rr == r_idx[i].item() and tt == t_idx[i].item() and hh != h_idx[i].item():
                        if hh < num_entities:
                            scores[hh] = float('-inf')
                true_score = scores[h_idx[i].item()]
                rank = (scores > true_score).sum().item() + 1
                head_ranks.append(rank)

    tail_ranks = torch.tensor(tail_ranks, dtype=torch.float32)
    head_ranks = torch.tensor(head_ranks, dtype=torch.float32)

    return {
        "tail_mrr": (1.0 / tail_ranks).mean().item(),
        "head_mrr": (1.0 / head_ranks).mean().item(),
        "filtered_mrr": ((1.0 / tail_ranks).mean().item() + (1.0 / head_ranks).mean().item()) / 2,
        "tail_hits@1": (tail_ranks <= 1).float().mean().item(),
        "tail_hits@3": (tail_ranks <= 3).float().mean().item(),
        "tail_hits@10": (tail_ranks <= 10).float().mean().item(),
        "head_hits@1": (head_ranks <= 1).float().mean().item(),
        "head_hits@3": (head_ranks <= 3).float().mean().item(),
        "head_hits@10": (head_ranks <= 10).float().mean().item(),
        "n_triples": n,
    }


def run_all_baselines(
    data, train_ei: Dict, val_ei: Dict, test_ei: Dict,
    dim: int = 128, num_epochs: int = 100, device: str = "cpu",
) -> dict:
    """Run all baseline models and return comparison results."""
    global_id_map, rel_map, num_entities = build_triple_mapping(
        train_ei, val_ei, test_ei, data
    )
    num_relations = len(rel_map)

    train_triples = edges_to_triples(train_ei, rel_map, global_id_map)
    val_triples = edges_to_triples(val_ei, rel_map, global_id_map)
    test_triples = edges_to_triples(test_ei, rel_map, global_id_map)

    # Build all-true set for filtered evaluation
    all_true_set = set()
    for triples in [train_triples, val_triples, test_triples]:
        for j in range(triples.shape[0]):
            all_true_set.add((
                int(triples[j, 0]), int(triples[j, 1]), int(triples[j, 2])
            ))

    results = {}

    model_classes = {
        "TransE": TransE,
        "DistMult": DistMult,
        "ComplEx": ComplEx,
        "RotatE": RotatE,
    }

    for name, cls in model_classes.items():
        print(f"\n{'='*50}")
        print(f"Training {name} (dim={dim}, epochs={num_epochs})")
        print(f"{'='*50}")

        model = cls(num_entities, num_relations, dim=dim)
        trainer = BaselineTrainer(model, learning_rate=1e-3,
                                  num_epochs=num_epochs, device=device)
        trainer.fit(train_triples, num_entities, verbose=True)

        metrics = evaluate_baseline_filtered(
            model, test_triples, all_true_set, num_entities
        )
        results[name] = {k: round(v, 4) for k, v in metrics.items()}
        print(f"  Filtered MRR: {metrics['filtered_mrr']:.4f} | "
              f"H@1: {metrics['tail_hits@1']:.4f}/{metrics['head_hits@1']:.4f} | "
              f"H@10: {metrics['tail_hits@10']:.4f}/{metrics['head_hits@10']:.4f}")

    return results
