#!/usr/bin/env python3
"""One-shot: PubMedBERT 768d -> PCA 128d + retrain + harvest."""
import torch, json, time, os, sys
from pathlib import Path
from datetime import datetime

CACHE = Path("checkpoints/pca_features/features_128d.pt")
CACHE.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(msg, flush=True)

def generate_features(data):
    """PubMedBERT encoding + PCA 768->128d, with checkpoint/resume."""
    if CACHE.exists():
        log(f"  Loading cached features from {CACHE}")
        return torch.load(CACHE, weights_only=False)

    log("  Building entity summaries (entity name fallback)...")
    summaries = {}
    for nt in data.node_types:
        summaries[nt] = {}
        if hasattr(data[nt], 'name'):
            for idx, name in enumerate(data[nt].name):
                summaries[nt][idx] = str(name) if name else nt
        else:
            for idx in range(data[nt].num_nodes):
                summaries[nt][idx] = f"{nt}_{idx}"
    total = sum(len(s) for s in summaries.values())
    log(f"  {total} summaries built")

    log("  Loading PubMedBERT...")
    from data.node_features import PubMedBERTEncoder
    encoder = PubMedBERTEncoder()
    log(f"  Encoding {total} entities (batch_size=64)...")
    pb_feats = {}
    for nt, idx_sums in summaries.items():
        n = data[nt].num_nodes
        texts = [idx_sums.get(i, nt) for i in range(n)]
        emb = encoder.encode_batch(texts, batch_size=64)
        pb_feats[nt] = emb
        log(f"    {nt}: {emb.shape}")
    del encoder

    log("  PCA 768d -> 128d...")
    from sklearn.decomposition import PCA
    pca_feats = {}
    for nt, emb in pb_feats.items():
        if emb.shape[0] < 128:
            pca_feats[nt] = torch.nn.functional.pad(emb, (0, 128 - emb.shape[1]))
            continue
        pca = PCA(n_components=128, random_state=42)
        reduced = pca.fit_transform(emb.numpy())
        pca_feats[nt] = torch.tensor(reduced, dtype=torch.float32)
        var = pca.explained_variance_ratio_.sum()
        log(f"    {nt}: {emb.shape} -> {pca_feats[nt].shape} (variance: {var:.3f})")

    torch.save(pca_feats, CACHE)
    log(f"  Saved to {CACHE}")
    return pca_feats


def train(data, features, output_dir="checkpoints/pca_features"):
    """Train TemperedHGT with PCA features."""
    log("\n  Training TemperedHGT 100 epochs...")

    for nt in data.node_types:
        data[nt].x = features[nt]

    ets = [(et, data[et].edge_index.shape[1]) for et in data.edge_types if et[1] != 'MENTIONED_IN']
    ets.sort(key=lambda x: -x[1])
    top20 = [et for et, n in ets[:20]]

    train_ei, val_ei, neg_ei = {}, {}, {}
    for et in top20:
        ei = data[et].edge_index; n = ei.shape[1]
        perm = torch.randperm(n); tn = int(n * 0.85)
        train_ei[et] = ei[:, perm[:tn]]
        if n >= 30:
            val_ei[et] = ei[:, perm[tn:tn + int(n * 0.1)]]
        neg_ei[et] = torch.stack([
            torch.randint(0, data[et[0]].num_nodes, (tn,)),
            torch.randint(0, data[et[2]].num_nodes, (tn,)),
        ])

    import yaml
    temp_init = {f'{s}__{r}__{d}': 1.0 for s, r, d in top20}
    cfg_path = Path("config/anchor_config.yaml")
    if cfg_path.exists():
        with open(cfg_path, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        for cat in cfg.get('temperature_init', {}).values():
            tau = cat.get('tau_init', 1.0)
            for rn in cat.get('relations', []):
                for s, r, d in top20:
                    if r == rn:
                        temp_init[f'{s}__{r}__{d}'] = tau

    from models.tempered_hgt import TemperedHGT
    model = TemperedHGT(
        in_channels={nt: 128 for nt in data.node_types},
        hidden_channels=128, out_channels=128, num_heads=4, num_layers=2,
        meta_relations=top20, temperature_init=temp_init,
    )
    log(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    from training.link_prediction import LinkPredictionTrainer
    trainer = LinkPredictionTrainer(
        model=model, learning_rate=5e-3, num_epochs=100, patience=20,
        device="cpu", checkpoint_dir=output_dir,
    )
    result = trainer.fit(data, train_ei, val_ei, neg_ei, verbose=True)
    return result, model


def main():
    t0 = time.time()
    log("=" * 55)
    log(f"PubMedBERT+PCA Pipeline  {datetime.now():%H:%M:%S}")
    log("=" * 55)

    log("\n[1/3] Loading data...")
    data = torch.load("data/processed/heterodata.pt", weights_only=False)
    log(f"  {sum(data[nt].num_nodes for nt in data.node_types):,} nodes, "
        f"{sum(data[et].edge_index.shape[1] for et in data.edge_types):,} edges")

    log("\n[2/3] Generating PCA features...")
    features = generate_features(data)

    log("\n[3/3] Training...")
    result, model = train(data, features)

    t = time.time() - t0; m, s = divmod(t, 60)
    best = result["history"][result["best_epoch"]]
    summary = {
        "timestamp": datetime.now().isoformat(),
        "features": "PubMedBERT 768d -> PCA 128d",
        "best_epoch": result["best_epoch"],
        "auroc": best.get("val_auroc", 0),
        "mrr": result["best_val_mrr"],
        "hits10": best.get("val_hits@10", 0),
        "time_s": t,
        "checkpoint": f"checkpoints/pca_features/checkpoint_epoch_{result['best_epoch']}.pt",
    }
    with open("checkpoints/pca_features/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nDone in {int(m)}m {int(s)}s")
    log(f"AUROC: {summary['auroc']:.4f} | MRR: {summary['mrr']:.4f} | Hits@10: {summary['hits10']:.4f}")
    return summary

if __name__ == "__main__":
    main()
