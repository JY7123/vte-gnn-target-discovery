# training/metrics.py
"""Link prediction evaluation metrics: AUROC, MRR, Hits@K, Filtered Ranking.

Two tiers of evaluation:
  Tier 1 (fast, per-epoch): balanced AUROC, simple MRR, simple Hits@K
    — used during training for early stopping
  Tier 2 (slow, final only): filtered MRR, filtered Hits@1/3/10
    — standard KG link prediction evaluation (Bordes et al. 2013)
    — per-relation-type breakdown
"""
import torch
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


def compute_auroc(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> float:
    """AUROC via ranking: fraction of (pos, neg) pairs where pos > neg.

    Uses the probabilistic interpretation of AUROC: for a randomly chosen
    positive and negative sample, the probability that the positive scores
    higher than the negative.

    Parameters
    ----------
    pos_logits : torch.Tensor
        1-D tensor of scores for positive edges.
    neg_logits : torch.Tensor
        1-D tensor of scores for negative edges.

    Returns
    -------
    float
        AUROC score in [0.0, 1.0]. Returns 0.5 when either input is empty.
    """
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return 0.5

    pos = pos_logits.unsqueeze(1)  # [N_pos, 1]
    neg = neg_logits.unsqueeze(0)  # [1, N_neg]
    comparisons = (pos > neg).float()  # [N_pos, N_neg]
    return comparisons.mean().item()


def compute_mrr(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> float:
    """Mean Reciprocal Rank.

    For each positive, computes its rank among all negatives (how many negatives
    score higher than the positive, plus one). MRR is the mean of the
    reciprocal of these ranks.

    Parameters
    ----------
    pos_logits : torch.Tensor
        1-D tensor of scores for positive edges.
    neg_logits : torch.Tensor
        1-D tensor of scores for negative edges.

    Returns
    -------
    float
        MRR score in [0.0, 1.0]. Returns 0.0 when pos_logits is empty.
    """
    if pos_logits.numel() == 0:
        return 0.0

    mrr_sum = 0.0
    for pos_score in pos_logits:
        rank = (neg_logits > pos_score).sum().item() + 1
        mrr_sum += 1.0 / rank

    return mrr_sum / pos_logits.numel()


def compute_hits_at_k(pos_logits: torch.Tensor, neg_logits: torch.Tensor,
                       k: int = 10) -> float:
    """Hits@K: fraction of positives ranked in the top-K.

    Works with both 1-D and 2-D inputs. For 1-D inputs, each positive is
    compared against the same pool of negatives. For 2-D inputs, each row of
    positives is compared against the corresponding row of negatives.

    Parameters
    ----------
    pos_logits : torch.Tensor
        1-D tensor [N_pos] or 2-D tensor [batch, N_pos_per_sample] of positive scores.
    neg_logits : torch.Tensor
        1-D tensor [N_neg] or 2-D tensor [batch, N_neg_per_sample] of negative scores.
    k : int
        The rank cutoff.

    Returns
    -------
    float
        Hits@K score in [0.0, 1.0].
    """
    if pos_logits.dim() == 1:
        pos_logits = pos_logits.unsqueeze(1)
    if neg_logits.dim() == 1:
        neg_logits = neg_logits.unsqueeze(0).expand(pos_logits.shape[0], -1)

    hits = 0
    total = 0
    for i in range(pos_logits.shape[0]):
        pos_scores = pos_logits[i]
        neg_scores = neg_logits[i] if neg_logits.dim() > 1 else neg_logits

        for pos_score in pos_scores:
            rank = (neg_scores > pos_score).sum().item() + 1
            if rank <= k:
                hits += 1
            total += 1

    return hits / total if total > 0 else 0.0


# ── Filtered Ranking Metrics (KG Standard, Tier 2) ──

def compute_filtered_ranks(
    z_dict: Dict[str, torch.Tensor],
    eval_triples: List[Tuple[int, int, int]],  # (head_idx, rel_key, tail_idx)
    src_type: str,
    dst_type: str,
    all_true_triples: Set[Tuple[int, int]],  # (head_idx, tail_idx) known true
    num_src: int,
    num_dst: int,
    decode_fn,
    batch_size: int = 1024,
) -> Dict[str, torch.Tensor]:
    """Compute filtered ranks for tail and head prediction.

    For each eval triple (h, r, t):
      - Tail prediction: rank (h, r, t) among all (h, r, t') for t' in dst_type
      - Head prediction: rank (h, r, t) among all (h', r, t) for h' in src_type

    Filtered setting: corrupted triples that appear in all_true_triples are
    excluded from ranking (their scores are set to -inf so they don't affect rank).

    Args:
        z_dict: Node embeddings {type: [N, dim]}
        eval_triples: List of (h, t) pairs for a single relation type
        src_type, dst_type: Node type names
        all_true_triples: Set of (h, t) known to be true in train/val/test
        num_src, num_dst: Number of source/destination entities
        decode_fn: function(z_dict, edge_index, src_t, dst_t) -> scores
        batch_size: Batch size for scoring to manage memory

    Returns:
        {"tail_ranks": [N], "head_ranks": [N], "tail_mrr": float, "head_mrr": float,
         "tail_hits@1": float, "tail_hits@3": float, "tail_hits@10": float, ...}
    """
    z_src = z_dict[src_type]  # [num_src, dim]
    z_dst = z_dict[dst_type]  # [num_dst, dim]

    # Pre-compute all-pair scores: [num_src, num_dst]
    all_scores = _batched_matmul(z_src, z_dst, batch_size)

    # Build per-head forbidden-tail and per-tail forbidden-head index sets
    # from all_true_triples. This avoids modifying the shared all_scores matrix.
    forbidden_tails = defaultdict(set)  # {h: {t1, t2, ...}}
    forbidden_heads = defaultdict(set)  # {t: {h1, h2, ...}}
    for h, t in all_true_triples:
        if h < num_src and t < num_dst:
            forbidden_tails[h].add(t)
            forbidden_heads[t].add(h)

    tail_ranks = []
    head_ranks = []

    for h, t in eval_triples:
        true_score = (z_src[h] * z_dst[t]).sum().item()

        # Tail prediction: rank among all (h, t')
        tail_scores = all_scores[h].clone()
        for t_forbidden in forbidden_tails.get(h, set()):
            if t_forbidden != t:
                tail_scores[t_forbidden] = float('-inf')
        tail_rank = (tail_scores > true_score).sum().item() + 1
        tail_ranks.append(tail_rank)

        # Head prediction: rank among all (h', t)
        head_scores = all_scores[:, t].clone()
        for h_forbidden in forbidden_heads.get(t, set()):
            if h_forbidden != h:
                head_scores[h_forbidden] = float('-inf')
        head_rank = (head_scores > true_score).sum().item() + 1
        head_ranks.append(head_rank)

    tail_ranks = torch.tensor(tail_ranks, dtype=torch.float32)
    head_ranks = torch.tensor(head_ranks, dtype=torch.float32)

    return {
        "tail_ranks": tail_ranks,
        "head_ranks": head_ranks,
        "tail_mrr": (1.0 / tail_ranks).mean().item(),
        "head_mrr": (1.0 / head_ranks).mean().item(),
        "tail_hits@1": (tail_ranks <= 1).float().mean().item(),
        "tail_hits@3": (tail_ranks <= 3).float().mean().item(),
        "tail_hits@10": (tail_ranks <= 10).float().mean().item(),
        "head_hits@1": (head_ranks <= 1).float().mean().item(),
        "head_hits@3": (head_ranks <= 3).float().mean().item(),
        "head_hits@10": (head_ranks <= 10).float().mean().item(),
    }


def _batched_matmul(A: torch.Tensor, B: torch.Tensor,
                    batch_size: int) -> torch.Tensor:
    """Memory-efficient batched matrix multiplication A @ B.T."""
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2
    result = torch.empty(M, N, dtype=A.dtype, device=A.device)
    for i in range(0, M, batch_size):
        end = min(i + batch_size, M)
        result[i:end] = A[i:end] @ B.T
    return result


def build_true_triples_set(
    edge_dicts: List[Dict[Tuple, torch.Tensor]],
    target_edge_types: Optional[List[Tuple]] = None,
) -> Dict[Tuple, Set[Tuple[int, int]]]:
    """Build set of known true (head, tail) pairs from train/val/test edges.

    Args:
        edge_dicts: list of edge dicts (e.g. [train_ei, val_ei, test_ei])
        target_edge_types: subset of edge types to include (None = all)

    Returns:
        {edge_type: set((h_idx, t_idx), ...)}
    """
    true_sets = defaultdict(set)
    for ei_dict in edge_dicts:
        for et, ei in ei_dict.items():
            if target_edge_types and et not in target_edge_types:
                continue
            for j in range(ei.shape[1]):
                true_sets[et].add((int(ei[0, j]), int(ei[1, j])))
    return dict(true_sets)


def filtered_evaluation(
    z_dict: Dict[str, torch.Tensor],
    eval_ei: Dict[Tuple, torch.Tensor],
    all_true_sets: Dict[Tuple, Set[Tuple[int, int]]],
    num_nodes_dict: Dict[str, int],
    decode_fn,
    batch_size: int = 1024,
) -> dict:
    """Run full filtered evaluation across all edge types.

    Returns:
        {
          "filtered_mrr": float,        # average of tail_mrr and head_mrr
          "filtered_hits@1": float,
          "filtered_hits@3": float,
          "filtered_hits@10": float,
          "per_relation": {edge_type_str: {mrr, hits@1, hits@3, hits@10, n_triples}},
          "n_triples_evaluated": int,
        }
    """
    total_tail_mrr = 0.0
    total_head_mrr = 0.0
    total_tail_h1 = 0.0
    total_tail_h3 = 0.0
    total_tail_h10 = 0.0
    total_head_h1 = 0.0
    total_head_h3 = 0.0
    total_head_h10 = 0.0
    total_n = 0
    per_relation = {}

    for et, ei in eval_ei.items():
        src_t, rel, dst_t = et
        num_src = num_nodes_dict.get(src_t, 0)
        num_dst = num_nodes_dict.get(dst_t, 0)
        if num_src == 0 or num_dst == 0:
            continue

        triples = [(int(ei[0, j]), int(ei[1, j])) for j in range(ei.shape[1])]
        if not triples:
            continue

        true_set = all_true_sets.get(et, set())

        rel_results = compute_filtered_ranks(
            z_dict=z_dict,
            eval_triples=triples,
            src_type=src_t,
            dst_type=dst_t,
            all_true_triples=true_set,
            num_src=num_src,
            num_dst=num_dst,
            decode_fn=decode_fn,
            batch_size=batch_size,
        )

        n_triples = len(triples)
        total_tail_mrr += rel_results["tail_mrr"] * n_triples
        total_head_mrr += rel_results["head_mrr"] * n_triples
        total_tail_h1 += rel_results["tail_hits@1"] * n_triples
        total_tail_h3 += rel_results["tail_hits@3"] * n_triples
        total_tail_h10 += rel_results["tail_hits@10"] * n_triples
        total_head_h1 += rel_results["head_hits@1"] * n_triples
        total_head_h3 += rel_results["head_hits@3"] * n_triples
        total_head_h10 += rel_results["head_hits@10"] * n_triples
        total_n += n_triples

        et_key = f"{src_t}__{rel}__{dst_t}"
        per_relation[et_key] = {
            "n_triples": n_triples,
            "tail_mrr": round(rel_results["tail_mrr"], 4),
            "head_mrr": round(rel_results["head_mrr"], 4),
            "mrr": round((rel_results["tail_mrr"] + rel_results["head_mrr"]) / 2, 4),
            "hits@1": round((rel_results["tail_hits@1"] + rel_results["head_hits@1"]) / 2, 4),
            "hits@3": round((rel_results["tail_hits@3"] + rel_results["head_hits@3"]) / 2, 4),
            "hits@10": round((rel_results["tail_hits@10"] + rel_results["head_hits@10"]) / 2, 4),
        }

    if total_n == 0:
        return {"filtered_mrr": 0.0, "filtered_hits@1": 0.0,
                "filtered_hits@3": 0.0, "filtered_hits@10": 0.0,
                "per_relation": {}, "n_triples_evaluated": 0}

    avg_tail_mrr = total_tail_mrr / total_n
    avg_head_mrr = total_head_mrr / total_n

    return {
        "filtered_mrr": round((avg_tail_mrr + avg_head_mrr) / 2, 4),
        "filtered_tail_mrr": round(avg_tail_mrr, 4),
        "filtered_head_mrr": round(avg_head_mrr, 4),
        "filtered_hits@1": round((total_tail_h1 + total_head_h1) / (2 * total_n), 4),
        "filtered_hits@3": round((total_tail_h3 + total_head_h3) / (2 * total_n), 4),
        "filtered_hits@10": round((total_tail_h10 + total_head_h10) / (2 * total_n), 4),
        "per_relation": per_relation,
        "n_triples_evaluated": total_n,
    }
