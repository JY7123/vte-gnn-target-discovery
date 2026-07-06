#!/usr/bin/env python3
"""Full training with PubMedBERT features + checkpoint/resume.

Estimated: ~7min encoding + ~5min Node2Vec + ~15min train = ~30min total.
"""
import sys, json, time, pickle, torch
from pathlib import Path
from datetime import datetime

CHECKPOINT_DIR = Path("checkpoints/full_training_v2")
FEAT_CACHE = CHECKPOINT_DIR / "features_cache.pt"

def log(msg):
    print(msg, flush=True)

def main():
    t0 = time.time()
    log("=" * 55)
    log(f"VTE GNN Full Training v2  {datetime.now():%H:%M:%S}")
    log(f"PubMedBERT(768d) + Node2Vec(128d) + TemperedHGT(128d)")
    log("=" * 55)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    log("\n[1/4] Loading Phase 1 data...")
    data = torch.load("data/processed/heterodata.pt", weights_only=False)
    train_ei = torch.load("data/processed/train_edges.pt", weights_only=False)
    neg_ei = torch.load("data/processed/negative_edges.pt", weights_only=False)
    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(ei.shape[1] for ei in train_ei.values())
    log(f"  {n_nodes} nodes, {n_edges} train edges, {len(data.edge_types)} edge types")

    # ── Features with checkpoint/resume ──
    node_types = list(data.node_types)
    num_nodes_dict = {nt: data[nt].num_nodes for nt in node_types}

    if FEAT_CACHE.exists():
        log(f"\n[2/4] RESUME: loading cached features from {FEAT_CACHE}")
        combined = torch.load(FEAT_CACHE, weights_only=False)
    else:
        log("\n[2/4] Generating features (PubMedBERT 768d + Node2Vec 128d)...")

        # Build summaries (entity name fallback)
        log("  Building entity summaries...")
        summaries = {}
        for nt in node_types:
            summaries[nt] = {}
            if hasattr(data[nt], 'name'):
                for idx, name in enumerate(data[nt].name):
                    summaries[nt][idx] = str(name) if name else f"{nt}_{idx}"
            else:
                for idx in range(data[nt].num_nodes):
                    summaries[nt][idx] = f"{nt}_{idx}"
        total = sum(len(s) for s in summaries.values())
        log(f"  {total} summaries built (entity name fallback)")

        # PubMedBERT encoding
        from data.node_features import NodeFeaturePipeline
        pipeline = NodeFeaturePipeline(
            pubmedbert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            node2vec_dim=128,
            output_dir=str(CHECKPOINT_DIR),
        )

        log("  Encoding with PubMedBERT (batch_size=64)...")
        t_enc = time.time()
        pb_feats = pipeline.generate_pubmedbert_features(summaries)
        enc_time = time.time() - t_enc
        total_emb = sum(f.shape[0] for f in pb_feats.values())
        log(f"  Encoded {total_emb} entities in {enc_time/60:.1f} min ({total_emb/enc_time:.0f} entities/s)")

        # Node2Vec
        log("  Training Node2Vec...")
        t_n2v = time.time()
        n2v_feats = pipeline.generate_node2vec_features(train_ei, num_nodes_dict)
        log(f"  Node2Vec done in {(time.time()-t_n2v)/60:.1f} min")

        # Combine + normalize
        combined = pipeline.combine_and_normalize(pb_feats, n2v_feats)
        torch.save(combined, FEAT_CACHE)
        log(f"  Features saved to {FEAT_CACHE} ({combined[next(iter(combined))].shape[-1]}d)")

    # Attach features to data
    for nt in node_types:
        data[nt].x = combined[nt]

    # ── Model ──
    log("\n[3/4] Building TemperedHGT(128d, 4 heads, 2 layers)...")
    meta_relations = list(data.edge_types)
    feat_dim = combined[next(iter(combined))].shape[-1]

    import yaml
    temp_init = {}
    for src, rel, dst in meta_relations:
        temp_init[f"{src}__{rel}__{dst}"] = 1.0
    # Load temperatures from config
    cfg_path = Path("config/anchor_config.yaml")
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for cat in cfg.get("temperature_init", {}).values():
            tau = cat.get("tau_init", 1.0)
            for rn in cat.get("relations", []):
                for src, rel, dst in meta_relations:
                    if rel == rn:
                        temp_init[f"{src}__{rel}__{dst}"] = tau

    from models.tempered_hgt import TemperedHGT
    model = TemperedHGT(
        in_channels={nt: feat_dim for nt in node_types},
        hidden_channels=128, out_channels=128,
        num_heads=4, num_layers=2,
        meta_relations=meta_relations,
        temperature_init=temp_init,
    )
    log(f"  {sum(p.numel() for p in model.parameters()):,} params, in_dim={feat_dim}")

    # ── Train ──
    log("\n[4/4] Training 100 epochs (cosine decay, early stop patience=10)...")
    from training.link_prediction import LinkPredictionTrainer
    trainer = LinkPredictionTrainer(
        model=model, learning_rate=5e-3, num_epochs=100, patience=15,
        device="cpu", checkpoint_dir=str(CHECKPOINT_DIR),
        batch_size=256, num_neighbors=[10, 5, 5],
    )
    result = trainer.fit(data, train_ei, train_ei, neg_ei, verbose=True)

    t = time.time() - t0
    m, s = divmod(t, 60)
    best = result["history"][result["best_epoch"]]
    log(f"\n{'='*55}")
    log(f"DONE in {int(m)}m {int(s)}s")
    log(f"Best epoch: {result['best_epoch']}  |  MRR: {result['best_val_mrr']:.4f}  |  AUROC: {best.get('val_auroc',0):.4f}  |  Hits@10: {best.get('val_hits@10',0):.4f}")
    log(f"{'='*55}")

    with open(CHECKPOINT_DIR / "summary.json", "w") as f:
        json.dump({**result, "time_s": t, "feat_dim": feat_dim,
                    "best_auroc": best.get('val_auroc', 0),
                    "best_hits10": best.get('val_hits@10', 0)}, f, indent=2)

if __name__ == "__main__":
    main()
