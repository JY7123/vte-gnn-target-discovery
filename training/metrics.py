# training/metrics.py
"""Link prediction evaluation metrics: AUROC, MRR, Hits@K."""
import torch


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
