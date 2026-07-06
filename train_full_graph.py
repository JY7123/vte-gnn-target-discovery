#!/usr/bin/env python3
"""Train TemperedHGT on FULL connected KG (82K nodes, 248K edges, 5056 edge types)."""
import sys, json, time, torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def log(msg):
    print(msg, flush=True)

def main():
    t0 = time.time()
    log("=" * 55)
    log(f"TRAINING ON FULL CONNECTED KG  {datetime.now():%H:%M:%S}")
    log(f"82K nodes | 248K edges | 5056 edge types | 100% connectivity")
    log("=" * 55)

    log("\n[1/3] Loading full HeteroData...")
    data = torch.load("data/processed/heterodata.pt", weights_only=False)
    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    log(f"  {n_nodes} nodes, {n_edges} edges, {len(data.edge_types)} edge types")

    # Build random features
    torch.manual_seed(42)
    for nt in data.node_types:
        data[nt].x = torch.randn(data[nt].num_nodes, 64) * 0.1

    # Filter: exclude MENTIONED_IN edges (literature citations, not biological)
    # and edge types with < 10 edges (too sparse for meaningful training)
    log("\n[2/3] Filtering edge types...")
    bio_edge_types = []
    for et in data.edge_types:
        src_t, rel, dst_t = et
        n = data[et].edge_index.shape[1]
        if rel == "MENTIONED_IN":
            continue  # skip literature citation edges
        if n < 10:
            continue  # skip tiny edge types
        bio_edge_types.append(et)
    log(f"  {len(bio_edge_types)} biological edge types kept (from {len(data.edge_types)} total)")

    # Build train/val/test + negatives
    train_ei = {}
    val_ei = {}
    neg_ei = {}

    for et in bio_edge_types:
        ei = data[et].edge_index
        n = ei.shape[1]
        perm = torch.randperm(n)
        train_n = int(n * 0.85)
        train_ei[et] = ei[:, perm[:train_n]]
        if n >= 30:
            val_n = int(n * 0.10)
            val_ei[et] = ei[:, perm[train_n:train_n+val_n]]

        src_t, rel, dst_t = et
        neg_src = torch.randint(0, data[src_t].num_nodes, (train_n,))
        neg_dst = torch.randint(0, data[dst_t].num_nodes, (train_n,))
        neg_ei[et] = torch.stack([neg_src, neg_dst])

    total_train = sum(ei.shape[1] for ei in train_ei.values())
    log(f"  Train: {total_train} edges, Val: {sum(ei.shape[1] for ei in val_ei.values())}")

    # Build model (only bio edge types)
    log("\n[3/3] Building TemperedHGT + training 100 epochs...")
    meta_relations = bio_edge_types
    temp_init = {f"{s}__{r}__{d}": 1.0 for s, r, d in meta_relations}

    from models.tempered_hgt import TemperedHGT
    model = TemperedHGT(
        in_channels={nt: 64 for nt in data.node_types},
        hidden_channels=64, out_channels=64,
        num_heads=4, num_layers=2,
        meta_relations=meta_relations, temperature_init=temp_init,
    )
    log(f"  {sum(p.numel() for p in model.parameters()):,} params")

    from training.link_prediction import LinkPredictionTrainer
    trainer = LinkPredictionTrainer(
        model=model, learning_rate=5e-3, num_epochs=100, patience=15,
        device="cpu", checkpoint_dir="checkpoints/full_graph",
        batch_size=256, num_neighbors=[10, 5, 5],
    )

    result = trainer.fit(data, train_ei, val_ei, neg_ei, verbose=True)

    t = time.time() - t0
    m, s = divmod(t, 60)
    best = result["history"][result["best_epoch"]]
    log(f"\n{'='*55}")
    log(f"DONE in {int(m)}m {int(s)}s")
    log(f"Best epoch: {result['best_epoch']}  |  MRR: {result['best_val_mrr']:.4f}  |  AUROC: {best.get('val_auroc',0):.4f}  |  Hits@10: {best.get('val_hits@10',0):.4f}")
    log(f"{'='*55}")

    with open("checkpoints/full_graph/summary.json", "w") as f:
        json.dump({**result, "time_s": t, "nodes": n_nodes, "edges": n_edges,
                    "edge_types": len(meta_relations)}, f, indent=2)

if __name__ == "__main__":
    main()
