#!/usr/bin/env python3
"""Quick training: Node2Vec features only, fast iteration."""
import sys, json, time, yaml, torch
from pathlib import Path
from datetime import datetime

def log(msg):
    print(msg, flush=True)

def main():
    t0 = time.time()
    log("=" * 60)
    log("VTE GNN Quick Training (Node2Vec features)")
    log(f"Start: {datetime.now():%H:%M:%S}")
    log("=" * 60)

    # Load data
    log("\n[1/3] Loading data...")
    data = torch.load("data/processed/heterodata.pt", weights_only=False)
    train_ei = torch.load("data/processed/train_edges.pt", weights_only=False)
    neg_ei = torch.load("data/processed/negative_edges.pt", weights_only=False)
    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(ei.shape[1] for ei in train_ei.values())
    log(f"  {n_nodes} nodes, {n_edges} train edges, {len(data.edge_types)} edge types")

    # Node2Vec features
    log("\n[2/3] Training Node2Vec...")
    from data.node_features import NodeFeaturePipeline
    pipeline = NodeFeaturePipeline(
        pubmedbert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        node2vec_dim=64, output_dir="data/processed",
    )
    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    n2v = pipeline.generate_node2vec_features(train_ei, num_nodes_dict)
    log(f"  Node2Vec done: {[(nt, n2v[nt].shape) for nt in sorted(n2v)][:5]}...")

    # Use Node2Vec features directly (skip PubMedBERT for speed)
    for nt in data.node_types:
        data[nt].x = n2v[nt]
    log(f"  Features: {data.node_types}")

    # Model
    meta_relations = list(data.edge_types)
    node_types = list(data.node_types)
    in_channels = {nt: 64 for nt in node_types}

    temp_init = {}
    for src, rel, dst in meta_relations:
        temp_init[f"{src}__{rel}__{dst}"] = 1.0

    from models.tempered_hgt import TemperedHGT
    model = TemperedHGT(
        in_channels=in_channels, hidden_channels=64, out_channels=64,
        num_heads=4, num_layers=2, meta_relations=meta_relations,
        temperature_init=temp_init,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model: {n_params:,} params, {len(meta_relations)} edge types")

    # Train
    log("\n[3/3] Training 100 epochs...")
    from training.link_prediction import LinkPredictionTrainer
    trainer = LinkPredictionTrainer(
        model=model, learning_rate=1e-3, num_epochs=100, patience=10,
        device="cpu", checkpoint_dir="checkpoints/quick_training",
        batch_size=256, num_neighbors=[10, 5, 5],
    )

    result = trainer.fit(data, train_ei, train_ei, neg_ei, verbose=True)

    total = time.time() - t0
    m, s = divmod(total, 60)
    log(f"\nDone in {int(m)}m {int(s)}s")
    log(f"Best epoch: {result['best_epoch']}, MRR: {result['best_val_mrr']:.4f}")

    with open("checkpoints/quick_training/training_summary.json", "w") as f:
        json.dump({"best_epoch": result["best_epoch"],
                    "best_val_mrr": result["best_val_mrr"],
                    "time_s": total}, f, indent=2)

if __name__ == "__main__":
    main()
