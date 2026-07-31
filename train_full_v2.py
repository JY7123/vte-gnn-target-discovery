#!/usr/bin/env python3
"""Full training with PubMedBERT features + proper train/val/test split + multi-seed.

Key fixes (addressing reviewer critique):
  1. Proper 80/10/10 stratified split — train/val/test are disjoint
  2. Val/test edges NEVER used for message passing
  3. Separate negative edges generated per split
  4. Node2Vec trained ONLY on train edges (per-seed, no structural leakage)
  5. PubMedBERT cached globally (text-based, independent of graph structure)
  6. Multi-seed runs (default 5) with mean +/- std reporting
  7. Commit hash + data hash saved for reproducibility

Estimated: ~7min encoding (once) + ~5min/seed Node2Vec + ~20min/seed train.
"""
import sys, json, time, hashlib, subprocess, shutil, torch
from pathlib import Path
from datetime import datetime

CHECKPOINT_DIR = Path("checkpoints/full_training_v2")
PB_CACHE = CHECKPOINT_DIR / "pubmedbert_cache.pt"  # text-based, no graph leakage
SPLIT_DIR = Path("data/processed")
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def log(msg):
    print(msg, flush=True)


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()[:8]
    except Exception:
        return "unknown"


def get_data_hash(data_path: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return "unknown"


def generate_or_load_pubmedbert(data, node_types: list) -> dict:
    """Generate (or load cached) PubMedBERT 768d features from entity names.

    These are purely text-based, independent of graph structure — no leakage risk.
    """
    if PB_CACHE.exists():
        log(f"  Loading cached PubMedBERT features from {PB_CACHE}")
        return torch.load(PB_CACHE, weights_only=False)

    log("  Building entity summaries from KG node names...")
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
    log(f"  {total} summaries built")

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
    log(f"  Encoded {total_emb} entities in {enc_time/60:.1f} min "
        f"({total_emb/enc_time:.0f} entities/s)")

    torch.save(pb_feats, PB_CACHE)
    log(f"  Saved to {PB_CACHE}")
    return pb_feats


def generate_node2vec_from_train(train_ei: dict, num_nodes_dict: dict,
                                  seed: int) -> dict:
    """Generate Node2Vec 128d features from TRAIN edges ONLY.

    Each seed gets its own Node2Vec trained exclusively on its train split.
    Zero leakage into val/test.
    """
    from data.node_features import NodeFeaturePipeline
    pipeline = NodeFeaturePipeline(
        pubmedbert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        node2vec_dim=128,
        output_dir=str(CHECKPOINT_DIR),
    )
    # Use seed for Node2Vec reproducibility
    torch.manual_seed(seed)
    return pipeline.generate_node2vec_features(train_ei, num_nodes_dict)


def combine_features(pb_feats: dict, n2v_feats: dict) -> dict:
    """Combine PubMedBERT + Node2Vec and L2-normalize."""
    from data.node_features import NodeFeaturePipeline
    pipeline = NodeFeaturePipeline(
        pubmedbert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        node2vec_dim=128,
        output_dir=str(CHECKPOINT_DIR),
    )
    return pipeline.combine_and_normalize(pb_feats, n2v_feats)


def generate_split_and_negatives(data, seed: int, edge_types: list = None):
    """Generate stratified split + per-split negative edges."""
    from data.temporal_split import RandomStratifiedSplitter
    from data.negative_sampling import NegativeSamplingPipeline

    splitter = RandomStratifiedSplitter(
        train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=seed,
        edge_types=edge_types,
    )
    train_ei, val_ei, test_ei = splitter.split(data)
    report = splitter.split_and_report(data)
    log(f"  Split: {report['train_edges']} train / {report['val_edges']} val "
        f"/ {report['test_edges']} test edges")

    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    neg_pipeline = NegativeSamplingPipeline({}, seed=seed)

    train_neg = neg_pipeline.degree_sampler.sample(
        train_ei, num_nodes_dict, num_negatives_per_edge=1)
    val_neg = neg_pipeline.degree_sampler.sample(
        val_ei, num_nodes_dict, num_negatives_per_edge=1)
    test_neg = neg_pipeline.degree_sampler.sample(
        test_ei, num_nodes_dict, num_negatives_per_edge=1)

    return train_ei, val_ei, test_ei, train_neg, val_neg, test_neg, report


def train_single_seed(data, pb_feats: dict, node_types: list,
                      num_nodes_dict: dict, seed: int,
                      meta_relations: list, temp_init: dict) -> dict:
    """Train with one random seed. Node2Vec trained only on this seed's train split."""
    seed_dir = CHECKPOINT_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    log(f"\n{'─'*50}")
    log(f"SEED {seed}")
    log(f"{'─'*50}")

    # 1. Split (seed-specific, only core edge types)
    train_ei, val_ei, test_ei, train_neg, val_neg, test_neg, split_report = \
        generate_split_and_negatives(data, seed, edge_types=meta_relations)

    # 2. Node2Vec from TRAIN edges only (no leakage into val/test)
    log("  Generating Node2Vec from TRAIN edges only...")
    t_n2v = time.time()
    n2v_feats = generate_node2vec_from_train(train_ei, num_nodes_dict, seed)
    log(f"  Node2Vec done in {(time.time()-t_n2v)/60:.1f} min")

    # 3. Combine features
    combined = combine_features(pb_feats, n2v_feats)
    feat_dim = combined[next(iter(combined))].shape[-1]
    for nt in node_types:
        data[nt].x = combined[nt]

    # 4. Save per-seed feature cache
    torch.save(combined, seed_dir / "features_cache.pt")

    # 5. Model
    from models.tempered_hgt import TemperedHGT
    model = TemperedHGT(
        in_channels={nt: feat_dim for nt in node_types},
        hidden_channels=128, out_channels=128,
        num_heads=4, num_layers=2,
        meta_relations=meta_relations,
        temperature_init=temp_init,
    )
    log(f"  {sum(p.numel() for p in model.parameters()):,} params, in_dim={feat_dim}")

    # 6. Train
    from training.link_prediction import LinkPredictionTrainer
    trainer = LinkPredictionTrainer(
        model=model, learning_rate=5e-3, num_epochs=100, patience=15,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir=str(seed_dir),
        batch_size=256, num_neighbors=[10, 5, 5],
    )

    result = trainer.fit(
        data=data,
        train_ei=train_ei,
        val_ei=val_ei,
        test_ei=test_ei,
        train_neg_ei=train_neg,
        val_neg_ei=val_neg,
        test_neg_ei=test_neg,
        verbose=True,
    )

    # 7. Save metadata
    with open(seed_dir / "split_report.json", "w") as f:
        json.dump(split_report, f, indent=2)
    torch.save(train_ei, seed_dir / "train_edges.pt")
    torch.save(val_ei, seed_dir / "val_edges.pt")
    torch.save(test_ei, seed_dir / "test_edges.pt")

    best_ckpt = seed_dir / f"checkpoint_epoch_{result['best_epoch']}.pt"
    if best_ckpt.exists():
        shutil.copy(best_ckpt, seed_dir / "checkpoint_best.pt")

    return {**result, "seed": seed, "split_report": split_report}


def main():
    t0 = time.time()
    git_hash = get_git_hash()
    data_hash = get_data_hash(SPLIT_DIR / "heterodata.pt")

    log("=" * 60)
    log(f"VTE GNN Full Training v2 (no-leakage)  {datetime.now():%H:%M:%S}")
    log(f"PubMedBERT(768d global) + Node2Vec(128d per-seed) + TemperedHGT(128d)")
    log(f"Git commit: {git_hash}  |  Data hash: {data_hash}")
    log(f"Seeds: {DEFAULT_SEEDS}")
    log("=" * 60)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load KG data ──
    log("\n[1/5] Loading KG data...")
    data = torch.load(SPLIT_DIR / "heterodata.pt", weights_only=False)
    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    log(f"  {n_nodes} nodes, {len(data.edge_types)} edge types")

    # Filter to node types with >0 nodes (skip e.g. Entity with 0 nodes)
    valid_nts = [nt for nt in data.node_types if data[nt].num_nodes > 0]
    num_nodes_dict = {nt: data[nt].num_nodes for nt in valid_nts}
    log(f"  {len(valid_nts)} valid node types (skipping {len(data.node_types) - len(valid_nts)} empty types)")

    # ── 2. PubMedBERT features (text-based, graph-independent, cached globally) ──
    log("\n[2/5] PubMedBERT features (text-based, no graph leakage)...")
    pb_feats = generate_or_load_pubmedbert(data, valid_nts)
    pb_dim = pb_feats[next(iter(pb_feats))].shape[-1]
    log(f"  PubMedBERT dimension: {pb_dim}d")

    # ── 3. Model config ──
    log("\n[3/5] Configuring TemperedHGT(128d, 4 heads, 2 layers)...")

    import yaml
    # Use edge types from anchor_config (35 core biomedical relations)
    # instead of all 5,056 Neo4j types for manageable training
    cfg_path = Path("config/anchor_config.yaml")
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        config_edge_types = [tuple(et) for et in cfg.get("edge_types", [])]
        # Keep only edge types present in data AND with valid node types
        meta_relations = [et for et in config_edge_types
                          if et in data.edge_types
                          and et[0] in valid_nts and et[2] in valid_nts]
        log(f"  Using {len(meta_relations)}/{len(config_edge_types)} config edge types")
    else:
        meta_relations = list(data.edge_types)
        log(f"  Using all {len(meta_relations)} edge types")

    temp_init = {}
    for src, rel, dst in meta_relations:
        temp_init[f"{src}__{rel}__{dst}"] = 1.0
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

    # ── 4. Multi-seed training (Node2Vec generated per-seed from train only) ──
    log("\n[4/5] Multi-seed training (Node2Vec per-seed from TRAIN only, no leakage)...")
    all_results = []
    for seed in DEFAULT_SEEDS:
        result = train_single_seed(
            data=data,
            pb_feats=pb_feats,
            node_types=valid_nts,
            num_nodes_dict=num_nodes_dict,
            seed=seed,
            meta_relations=meta_relations,
            temp_init=temp_init,
        )
        all_results.append(result)

    # ── 5. Aggregate ──
    log(f"\n[5/5] Aggregating results across {len(all_results)} seeds...")

    def _mean_std(values):
        t = torch.tensor(values, dtype=torch.float32)
        return float(t.mean()), float(t.std())

    val_aurocs = [r["history"][r["best_epoch"]].get("val_auroc", 0) for r in all_results]
    val_mrrs = [r["best_val_mrr"] for r in all_results]
    test_aurocs = [r.get("test_metrics", {}).get("auroc", 0) for r in all_results]
    test_mrrs = [r.get("test_metrics", {}).get("mrr", 0) for r in all_results]
    test_hits10 = [r.get("test_metrics", {}).get("hits@10", 0) for r in all_results]

    summary = {
        "git_hash": git_hash,
        "data_hash": data_hash,
        "n_seeds": len(all_results),
        "feat_dim": pb_dim + 128,
        "val_auroc_mean": round(_mean_std(val_aurocs)[0], 4),
        "val_auroc_std": round(_mean_std(val_aurocs)[1], 4),
        "val_mrr_mean": round(_mean_std(val_mrrs)[0], 4),
        "val_mrr_std": round(_mean_std(val_mrrs)[1], 4),
        "test_auroc_mean": round(_mean_std(test_aurocs)[0], 4),
        "test_auroc_std": round(_mean_std(test_aurocs)[1], 4),
        "test_mrr_mean": round(_mean_std(test_mrrs)[0], 4),
        "test_mrr_std": round(_mean_std(test_mrrs)[1], 4),
        "test_hits10_mean": round(_mean_std(test_hits10)[0], 4),
        "test_hits10_std": round(_mean_std(test_hits10)[1], 4),
        "per_seed": [
            {
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "val_mrr": round(r["best_val_mrr"], 4),
                "test_auroc": round(r.get("test_metrics", {}).get("auroc", 0), 4),
                "test_mrr": round(r.get("test_metrics", {}).get("mrr", 0), 4),
                "test_hits10": round(r.get("test_metrics", {}).get("hits@10", 0), 4),
            }
            for r in all_results
        ],
    }

    t = time.time() - t0
    m, s = divmod(t, 60)
    summary["total_time_min"] = round(t / 60, 1)

    log(f"\n{'='*60}")
    log(f"DONE in {int(m)}m {int(s)}s")
    log(f"Val MRR:   {summary['val_mrr_mean']:.4f} +/- {summary['val_mrr_std']:.4f}")
    log(f"Test AUROC: {summary['test_auroc_mean']:.4f} +/- {summary['test_auroc_std']:.4f}")
    log(f"Test MRR:   {summary['test_mrr_mean']:.4f} +/- {summary['test_mrr_std']:.4f}")
    log(f"Test H@10:  {summary['test_hits10_mean']:.4f} +/- {summary['test_hits10_std']:.4f}")
    log(f"{'='*60}")

    with open(CHECKPOINT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"Summary saved to {CHECKPOINT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
