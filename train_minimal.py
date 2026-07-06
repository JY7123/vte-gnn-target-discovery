#!/usr/bin/env python3
"""Minimal training: Node2Vec only, ZERO transformers dependency."""
import sys, json, time, torch, networkx as nx
from pathlib import Path
from datetime import datetime

def log(msg):
    print(msg, flush=True)

def main():
    t0 = time.time()
    log("=" * 50)
    log("VTE GNN Training (Node2Vec structural features)")
    log(f"Start: {datetime.now():%H:%M:%S}")
    log("=" * 50)

    # Load data
    log("\n[1/3] Loading data...")
    data = torch.load("data/processed/heterodata.pt", weights_only=False)
    train_ei = torch.load("data/processed/train_edges.pt", weights_only=False)
    neg_ei = torch.load("data/processed/negative_edges.pt", weights_only=False)
    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(ei.shape[1] for ei in train_ei.values())
    log(f"  {n_nodes} nodes, {n_edges} train edges")

    # Node2Vec directly (no NodeFeaturePipeline = no PubMedBERT load)
    log("\n[2/3] Training Node2Vec...")
    from node2vec import Node2Vec

    # Build combined graph with offsets
    node_types = list(data.node_types)
    num_nodes_dict = {nt: data[nt].num_nodes for nt in node_types}
    offset_map = {}
    offset = 0
    for nt in sorted(node_types):
        offset_map[nt] = offset
        offset += num_nodes_dict[nt]

    edges = []
    for (src_t, _, dst_t), ei in train_ei.items():
        so = offset_map[src_t]
        do = offset_map[dst_t]
        for i in range(ei.shape[1]):
            edges.append((int(ei[0,i])+so, int(ei[1,i])+do))

    G = nx.Graph()
    G.add_edges_from(edges)
    log(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    n2v = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4, quiet=True)
    model_n2v = n2v.fit(window=10, min_count=1, batch_words=4)
    log(f"  Node2Vec trained")

    # Extract per-type features
    total = sum(num_nodes_dict.values())
    all_emb = torch.zeros(total, 64)
    for i in range(total):
        all_emb[i] = torch.tensor(model_n2v.wv[i])

    features = {}
    for nt in sorted(node_types):
        start = offset_map[nt]
        n = num_nodes_dict[nt]
        features[nt] = all_emb[start:start+n]
        log(f"  {nt}: {features[nt].shape}")

    for nt in data.node_types:
        data[nt].x = features[nt]

    # Model
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
    log(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

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
