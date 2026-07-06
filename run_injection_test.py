#!/usr/bin/env python3
"""Standalone Injection Test: CRP→VTE and Ferritin→DVT.

Trains Tempered HGT vs Pure HGT on identical data + injected false-positive edges,
then compares prediction scores for the injected edges.

Usage: python run_injection_test.py
"""
import torch
import torch.nn as nn
import sys, time, json, copy
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.tempered_hgt import TemperedHGT
from training.edge_bias import CosineAnnealingDecay, EdgeBiasInitializer
from training.metrics import compute_auroc, compute_mrr, compute_hits_at_k

# ── Config ──────────────────────────────────────────────────────
NUM_EPOCHS = 50
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
LR = 5e-3
PATIENCE = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PCA_FEAT_PATH = "checkpoints/pca_features/features_128d.pt"
ANCHOR_CONFIG = "config/anchor_config.yaml"

# New negative control injection pairs
INJECTION_PAIRS = [
    {"source": "CRP", "source_type": "Protein",
     "target": "venous thromboembolism", "target_type": "Disease",
     "relation": "ASSOCIATED_WITH"},
    {"source": "ferritin", "source_type": "Protein",
     "target": "deep vein thrombosis", "target_type": "Disease",
     "relation": "ASSOCIATED_WITH"},
]

CHECKPOINT_DIR = Path("checkpoints/injection_test")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(msg, flush=True)


def find_idx(data, node_type: str, name: str) -> int:
    """Case-insensitive node lookup. Raises ValueError if not found."""
    names = [n.lower() for n in data[node_type].name]
    return names.index(name.lower())


def inject_edges(train_ei, injection_pairs, data):
    """Add injected edges to training edge_index dict. Returns (updated, injected_info)."""
    updated = {k: v.clone() for k, v in train_ei.items()}
    injected = []
    for pair in injection_pairs:
        src_idx = find_idx(data, pair["source_type"], pair["source"])
        dst_idx = find_idx(data, pair["target_type"], pair["target"])
        et = (pair["source_type"], pair["relation"], pair["target_type"])
        new_edge = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
        if et in updated:
            updated[et] = torch.cat([updated[et], new_edge], dim=1)
        else:
            updated[et] = new_edge
        injected.append((et, src_idx, dst_idx))
        log(f"  Injected: {pair['source']}({src_idx}) -> {pair['target']}({dst_idx}) [{et}]")
    return updated, injected


def build_temp_init(meta_relations, tau_value=1.0):
    """Build temperature init dict for all edge types."""
    return {f"{src}__{rel}__{dst}": tau_value for src, rel, dst in meta_relations}


def build_edge_bias(data, train_ei):
    """Build edge_weight_bias dict from anchor_config."""
    import yaml
    with open(ANCHOR_CONFIG) as f:
        cfg = yaml.safe_load(f)
    initializer = EdgeBiasInitializer(cfg.get("hard_priors", {}))
    node_name_to_idx = {}
    for nt in data.node_types:
        if hasattr(data[nt], 'name'):
            node_name_to_idx[nt] = {n: i for i, n in enumerate(data[nt].name)}
    return initializer.build(node_name_to_idx, list(train_ei.keys()))


def train_model(model, data, train_ei, neg_ei, x_dict,
                num_epochs=NUM_EPOCHS, lr=LR, patience=PATIENCE,
                edge_weight_bias=None, use_cos_decay=True,
                tag="model"):
    """Train a model, return (best_state_dict, metrics_history)."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    cos_scheduler = CosineAnnealingDecay(total_steps=num_epochs) if use_cos_decay else None

    # Move data to device
    x_dict_dev = {k: v.to(DEVICE) for k, v in x_dict.items()}
    train_ei_dev = {k: v.to(DEVICE) for k, v in train_ei.items()}
    neg_ei_dev = {k: v.to(DEVICE) for k, v in neg_ei.items()}
    ewb_dev = None
    if edge_weight_bias:
        ewb_dev = {k: v.to(DEVICE) for k, v in edge_weight_bias.items()}

    best_val_mrr = 0.0
    best_state = None
    best_epoch = 0
    patience_cnt = 0
    history = []

    t0 = time.time()
    for epoch in range(num_epochs):
        model.train()
        cd = cos_scheduler(epoch) if cos_scheduler else 0.0

        z_dict = model(x_dict_dev, train_ei_dev, cos_decay=cd, edge_weight_bias=ewb_dev)

        total_loss = 0.0
        n_edges = 0
        for et in train_ei_dev:
            if et not in neg_ei_dev:
                continue
            src_t, rel, dst_t = et
            pos_logits = model.decode(z_dict, train_ei_dev[et], src_t, dst_t)
            neg_logits = model.decode(z_dict, neg_ei_dev[et], src_t, dst_t)
            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
            loss = loss_fn(logits, labels)
            total_loss += loss
            n_edges += 1

        if n_edges == 0:
            continue
        total_loss = total_loss / n_edges

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                z_val = model(x_dict_dev, train_ei_dev, cos_decay=0.0, edge_weight_bias=ewb_dev)

                val_pos_scores = []
                val_neg_scores = []
                for et in train_ei_dev:
                    if et not in neg_ei_dev:
                        continue
                    src_t, rel, dst_t = et
                    pos_s = model.decode(z_val, train_ei_dev[et], src_t, dst_t)
                    neg_s = model.decode(z_val, neg_ei_dev[et], src_t, dst_t)
                    val_pos_scores.append(pos_s)
                    val_neg_scores.append(neg_s)

                if val_pos_scores:
                    pos_cat = torch.cat(val_pos_scores)
                    neg_cat = torch.cat(val_neg_scores)
                    auroc = compute_auroc(pos_cat, neg_cat)
                    mrr = compute_mrr(pos_cat, neg_cat)
                    history.append({"epoch": epoch, "auroc": auroc, "mrr": mrr, "loss": total_loss.item()})

                    if mrr > best_val_mrr:
                        best_val_mrr = mrr
                        best_epoch = epoch
                        best_state = copy.deepcopy(model.state_dict())
                        patience_cnt = 0
                    else:
                        patience_cnt += 1

                    elapsed = time.time() - t0
                    log(f"  [{tag}] Epoch {epoch:3d} | AUROC={auroc:.4f} MRR={mrr:.4f} Loss={total_loss.item():.4f} | {elapsed:.0f}s")

        if patience_cnt >= patience:
            log(f"  [{tag}] Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t0
    log(f"  [{tag}] Best: epoch={best_epoch}, MRR={best_val_mrr:.4f} | {elapsed:.0f}s total")
    return best_state, history, best_epoch


def score_injected(model, data, train_ei, x_dict, injected_edges, edge_weight_bias=None):
    """Score injected edges using trained model."""
    model.eval()
    x_dict_dev = {k: v.to(DEVICE) for k, v in x_dict.items()}
    train_ei_dev = {k: v.to(DEVICE) for k, v in train_ei.items()}
    ewb_dev = None
    if edge_weight_bias:
        ewb_dev = {k: v.to(DEVICE) for k, v in edge_weight_bias.items()}

    with torch.no_grad():
        z_dict = model(x_dict_dev, train_ei_dev, cos_decay=0.0, edge_weight_bias=ewb_dev)
        scores = []
        for et, src_idx, dst_idx in injected_edges:
            edge = torch.tensor([[src_idx], [dst_idx]], device=DEVICE)
            src_t, rel, dst_t = et
            score = model.decode(z_dict, edge, src_t, dst_t).item()
            scores.append(score)
    return scores


def main():
    log("=" * 60)
    log(f"VTE GNN Injection Test — {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Device: {DEVICE}")
    log("=" * 60)

    # ── Load data ──
    log("\n[1] Loading data...")
    data = torch.load("data/processed/heterodata.pt", weights_only=False)
    train_ei = torch.load("data/processed/train_edges.pt", weights_only=False)
    neg_ei = torch.load("data/processed/negative_edges.pt", weights_only=False)
    log(f"  Nodes: {sum(data[nt].num_nodes for nt in data.node_types):,}")
    log(f"  Train edges: {sum(ei.shape[1] for ei in train_ei.values()):,}")

    # ── Load PCA features ──
    log("\n[2] Loading PCA 128d features...")
    pca_feats = torch.load(PCA_FEAT_PATH, weights_only=False)
    for nt in data.node_types:
        if nt in pca_feats and pca_feats[nt].shape[0] == data[nt].num_nodes:
            data[nt].x = pca_feats[nt]
        else:
            data[nt].x = torch.randn(data[nt].num_nodes, HIDDEN_DIM) * 0.01
    log(f"  PCA features assigned to {len(pca_feats)} node types")

    # ── Inject false-positive edges ──
    log("\n[3] Injecting false-positive edges...")
    injected_train_ei, injected_info = inject_edges(train_ei, INJECTION_PAIRS, data)

    # ── Build shared config ──
    meta_relations = list(data.edge_types)
    in_channels = {nt: HIDDEN_DIM for nt in data.node_types}
    x_dict = {nt: data[nt].x for nt in data.node_types}

    # Build edge bias for Tempered HGT
    log("\n[4] Building edge bias from anchor config...")
    try:
        edge_bias = build_edge_bias(data, injected_train_ei)
        log(f"  Built edge bias for {len(edge_bias)} edge types")
    except Exception as e:
        log(f"  WARNING: Could not build edge bias ({e}), using None")
        edge_bias = None

    # ── Train Tempered HGT (full prior) ──
    log("\n[5] Training Tempered HGT (full prior: tau trainable + cos_decay + edge_bias)...")
    temp_init = build_temp_init(meta_relations, tau_value=1.0)
    tempered = TemperedHGT(
        in_channels=in_channels, hidden_channels=HIDDEN_DIM, out_channels=HIDDEN_DIM,
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS, meta_relations=meta_relations,
        temperature_init=temp_init,
    )
    tempered_state, tempered_hist, tempered_best = train_model(
        tempered, data, injected_train_ei, neg_ei, x_dict,
        edge_weight_bias=edge_bias, use_cos_decay=True, tag="Tempered"
    )

    # ── Train Pure HGT (no prior) ──
    log("\n[6] Training Pure HGT (no prior: tau≡1.0 frozen + cos_decay≡0 + no edge_bias)...")
    pure_temp_init = build_temp_init(meta_relations, tau_value=1.0)
    pure = TemperedHGT(
        in_channels=in_channels, hidden_channels=HIDDEN_DIM, out_channels=HIDDEN_DIM,
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS, meta_relations=meta_relations,
        temperature_init=pure_temp_init,
    )
    # Freeze tau values at 1.0 (tau is in each conv layer's .temperatures ParameterDict)
    for conv in pure.convs:
        for key in conv.temperatures._keys:
            conv.temperatures[key].requires_grad = False

    pure_state, pure_hist, pure_best = train_model(
        pure, data, injected_train_ei, neg_ei, x_dict,
        edge_weight_bias=None, use_cos_decay=False, tag="Pure"
    )

    # ── Score injected edges ──
    log("\n[7] Scoring injected edges...")
    tempered.load_state_dict(tempered_state)
    pure.load_state_dict(pure_state)

    tempered_scores = score_injected(tempered, data, injected_train_ei, x_dict,
                                     injected_info, edge_weight_bias=edge_bias)
    pure_scores = score_injected(pure, data, injected_train_ei, x_dict,
                                 injected_info, edge_weight_bias=None)

    # ── Report ──
    log("\n" + "=" * 60)
    log("INJECTION TEST RESULTS")
    log("=" * 60)

    results = []
    for i, (pair, (et, src_idx, dst_idx)) in enumerate(zip(INJECTION_PAIRS, injected_info)):
        src_name = pair["source"]
        dst_name = pair["target"]
        ts = tempered_scores[i]
        ps = pure_scores[i]
        delta = ts - ps
        if ps != 0:
            suppression_pct = (ps - ts) / abs(ps) * 100
        else:
            suppression_pct = 0

        log(f"  {src_name} -> {dst_name}:")
        log(f"    Tempered HGT: {ts:.4f}")
        log(f"    Pure HGT:     {ps:.4f}")
        log(f"    Delta:        {delta:+.4f}")
        log(f"    Suppression:  {suppression_pct:+.1f}%")
        results.append({
            "pair": f"{src_name} -> {dst_name}",
            "tempered_score": ts,
            "pure_score": ps,
            "delta": delta,
            "suppression_pct": suppression_pct,
        })

    # Save results
    output = {
        "date": datetime.now().isoformat(),
        "device": DEVICE,
        "config": {
            "num_epochs": NUM_EPOCHS, "hidden_dim": HIDDEN_DIM,
            "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS,
        },
        "injection_pairs": INJECTION_PAIRS,
        "results": results,
        "tempered_best_epoch": tempered_best,
        "pure_best_epoch": pure_best,
        "tempered_history": tempered_hist,
        "pure_history": pure_hist,
    }
    out_path = CHECKPOINT_DIR / "injection_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved to {out_path}")

    # Also save model checkpoints
    torch.save(tempered_state, CHECKPOINT_DIR / "tempered_best.pt")
    torch.save(pure_state, CHECKPOINT_DIR / "pure_best.pt")

    log("\nDone.")


if __name__ == "__main__":
    main()
