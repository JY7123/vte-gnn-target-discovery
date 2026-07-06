#!/usr/bin/env python3
"""Full training pipeline: features → train → checkpoint.

Phase 1 recap: features, Phase 2: train TemperedHGT 100 epochs.
Saves best checkpoint for harvest_figures.py to consume.
"""
import sys, json, time, yaml
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

def main():
    t0 = time.time()
    print("=" * 60)
    print("VTE GNN Full Training Pipeline")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Load Phase 1 data ──
    data_dir = Path("data/processed")
    print("\n[1/4] Loading Phase 1 data...")
    data = torch.load(data_dir / "heterodata.pt", weights_only=False)
    train_ei = torch.load(data_dir / "train_edges.pt", weights_only=False)
    neg_ei = torch.load(data_dir / "negative_edges.pt", weights_only=False)
    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(ei.shape[1] for ei in train_ei.values())
    print(f"  {n_nodes} nodes, {n_edges} train edges, {len(data.edge_types)} edge types")

    # ── Generate features ──
    print("\n[2/4] Generating node features...")

    # Build entity summaries (entity name as fallback — fast path)
    summaries = {}
    for nt in data.node_types:
        summaries[nt] = {}
        if hasattr(data[nt], 'name'):
            for idx, name in enumerate(data[nt].name):
                summaries[nt][idx] = str(name) if name else nt
        else:
            for idx in range(data[nt].num_nodes):
                summaries[nt][idx] = f"{nt}_{idx}"

    total_entities = sum(len(s) for s in summaries.values())
    print(f"  Built summaries for {total_entities} entities (name fallback)")

    # PubMedBERT encoding + Node2Vec via unified pipeline
    from data.node_features import NodeFeaturePipeline
    pipeline = NodeFeaturePipeline(
        pubmedbert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        node2vec_dim=128,
        output_dir="data/processed",
    )
    print(f"  Encoding {total_entities} entities (batch_size=64)...")
    pubmedbert_feats = pipeline.generate_pubmedbert_features(summaries)

    print("  Training Node2Vec on train edges...")
    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    n2v_features = pipeline.generate_node2vec_features(train_ei, num_nodes_dict)

    # Combine + normalize
    combined = pipeline.combine_and_normalize(pubmedbert_feats, n2v_features)
    pipeline.save_features(combined, "train")

    # Attach features to HeteroData
    for nt in data.node_types:
        data[nt].x = combined[nt]

    print(f"  Features attached: {[(nt, combined[nt].shape) for nt in sorted(data.node_types)]}")

    # ── Build model ──
    print("\n[3/4] Building TemperedHGT model...")

    meta_relations = list(data.edge_types)
    node_types = list(data.node_types)

    # Build in_channels from actual features
    in_channels = {}
    for nt in node_types:
        in_channels[nt] = combined[nt].shape[-1]

    # Temperature init from config
    temp_init = {}
    cfg_path = Path("config/anchor_config.yaml")
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        temp_cfg = cfg.get("temperature_init", {})
        for category in temp_cfg.values():
            tau = category.get("tau_init", 1.0)
            for rel_name in category.get("relations", []):
                for src, rel, dst in meta_relations:
                    if rel == rel_name:
                        temp_init[f"{src}__{rel}__{dst}"] = tau

    for src, rel, dst in meta_relations:
        key = f"{src}__{rel}__{dst}"
        if key not in temp_init:
            temp_init[key] = 1.0

    # Edge bias from config
    from training.edge_bias import EdgeBiasInitializer
    try:
        initializer = EdgeBiasInitializer.from_yaml("config/anchor_config.yaml")
        node_name_to_idx = {}
        for nt in node_types:
            if hasattr(data[nt], 'name'):
                node_name_to_idx[nt] = {name: idx for idx, name in enumerate(data[nt].name)}
            else:
                node_name_to_idx[nt] = {}
        bias_dict = initializer.build(node_name_to_idx, edge_types=meta_relations)
        edge_weight_bias = {}
        for et, bias_list in bias_dict.items():
            if et not in data.edge_types:
                continue
            ei = data[et].edge_index
            bias_tensor = torch.zeros(ei.shape[1])
            for src_idx, dst_idx, multiplier in bias_list:
                mask = (ei[0] == src_idx) & (ei[1] == dst_idx)
                bias_tensor[mask] = multiplier
            if bias_tensor.sum() > 0:
                edge_weight_bias[et] = bias_tensor
        print(f"  Edge bias: {len(edge_weight_bias)} edge types with configured priors")
    except Exception as e:
        print(f"  Edge bias skipped: {e}")
        edge_weight_bias = None

    from models.tempered_hgt import TemperedHGT
    model = TemperedHGT(
        in_channels=in_channels,
        hidden_channels=128,
        out_channels=128,
        num_heads=4,
        num_layers=2,
        meta_relations=meta_relations,
        temperature_init=temp_init,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters, {len(meta_relations)} edge types")

    # ── Train ──
    print("\n[4/4] Training TemperedHGT (100 epochs)...")

    from training.link_prediction import LinkPredictionTrainer

    trainer = LinkPredictionTrainer(
        model=model,
        learning_rate=1e-3,
        num_epochs=100,
        patience=10,
        device="cpu",
        checkpoint_dir="checkpoints/full_training",
        edge_weight_bias=edge_weight_bias,
        batch_size=256,
        num_neighbors=[10, 5, 5],
    )

    result = trainer.fit(data, train_ei, train_ei, neg_ei, verbose=True)

    # ── Save final artifacts ──
    total_time = time.time() - t0
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_time": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        "model_params": n_params,
        "best_epoch": result["best_epoch"],
        "best_val_mrr": result["best_val_mrr"],
        "best_val_auroc": result["history"][result["best_epoch"]].get("val_auroc", 0),
        "nodes": n_nodes,
        "train_edges": n_edges,
        "edge_types": len(meta_relations),
        "feature_dim": combined[next(iter(combined))].shape[-1],
    }

    with open("checkpoints/full_training/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val MRR: {result['best_val_mrr']:.4f}")
    print(f"  Best val AUROC: {summary['best_val_auroc']:.4f}")
    print(f"  Checkpoint: checkpoints/full_training/checkpoint_epoch_{result['best_epoch']}.pt")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    main()
