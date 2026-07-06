#!/usr/bin/env python3
"""Fastest possible training: random features, train immediately."""
import sys, json, time, torch
from pathlib import Path
from datetime import datetime

def log(msg):
    print(msg, flush=True)

def main():
    t0 = time.time()
    log("=" * 50)
    log(f"VTE GNN Fast Training  {datetime.now():%H:%M:%S}")
    log("=" * 50)

    log("\n[1/2] Loading data + building features...")
    data = torch.load("data/processed/heterodata.pt", weights_only=False)
    train_ei = torch.load("data/processed/train_edges.pt", weights_only=False)
    neg_ei = torch.load("data/processed/negative_edges.pt", weights_only=False)

    # Random features: seed so results are reproducible
    torch.manual_seed(42)
    for nt in data.node_types:
        data[nt].x = torch.randn(data[nt].num_nodes, 64) * 0.1

    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(ei.shape[1] for ei in train_ei.values())
    log(f"  {n_nodes} nodes, {n_edges} train edges, {len(data.edge_types)} edge types")
    log(f"  Random 64d features (seed=42)")

    log("\n[2/2] Building model + training 50 epochs...")
    meta_relations = list(data.edge_types)
    in_channels = {nt: 64 for nt in data.node_types}
    temp_init = {}
    for src, rel, dst in meta_relations:
        temp_init[f"{src}__{rel}__{dst}"] = 1.0

    from models.tempered_hgt import TemperedHGT
    model = TemperedHGT(
        in_channels=in_channels, hidden_channels=64, out_channels=64,
        num_heads=4, num_layers=2, meta_relations=meta_relations,
        temperature_init=temp_init,
    )
    log(f"  {sum(p.numel() for p in model.parameters()):,} params")

    from training.link_prediction import LinkPredictionTrainer
    trainer = LinkPredictionTrainer(
        model=model, learning_rate=5e-3, num_epochs=50, patience=10,
        device="cpu", checkpoint_dir="checkpoints/fast_training",
        batch_size=256, num_neighbors=[10, 5, 5],
    )

    result = trainer.fit(data, train_ei, train_ei, neg_ei, verbose=True)

    t = time.time() - t0
    m, s = divmod(t, 60)
    log(f"\nDone in {int(m)}m {int(s)}s  |  best_epoch={result['best_epoch']}  MRR={result['best_val_mrr']:.4f}  AUROC={result['history'][result['best_epoch']].get('val_auroc',0):.4f}")

    with open("checkpoints/fast_training/summary.json", "w") as f:
        json.dump({"best_epoch": result["best_epoch"], "best_val_mrr": result["best_val_mrr"], "time_s": t}, f, indent=2)

if __name__ == "__main__":
    main()
