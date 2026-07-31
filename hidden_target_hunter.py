#!/usr/bin/env python3
"""Hidden Target Hunter v3: Unconditioned global ranking with entity resolution.

Key properties:
  1. No per-source topk truncation — scores ALL candidate pairs globally
  2. Entity resolution: aggregates fragmented PAR-2/F2RL1 entities
  3. SAME features + checkpoint as the no-leakage training pipeline
  4. Global ranking by GNN score — NO anchor BFS gate
  5. Memory-safe: flattened topk + size-bounded heap
"""
import torch, json, time, math, os, heapq
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

# ── Entity resolution: fragment name patterns → canonical gene symbol ──
ENTITY_RESOLUTION = {
    "f2rl1": "F2RL1 (PAR-2)",
    "par-2": "F2RL1 (PAR-2)",
    "par2": "F2RL1 (PAR-2)",
    "proteinase-activated receptor 2": "F2RL1 (PAR-2)",
    "protease-activated receptor 2": "F2RL1 (PAR-2)",
    "coagulation factor ii receptor-like 1": "F2RL1 (PAR-2)",
}

# ── Background annotation sets (for post-hoc labeling, NOT filtering) ──
TEXTBOOK_VTE_GENES = {
    'f2', 'prothrombin', 'thrombin', 'f5', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
    'f13', 'tf', 'tissue factor', 'vwd', 'vwf', 'von willebrand', 'adamts13',
    'protein c', 'protein s', 'antithrombin', 'serpine1', 'pai-1', 'tfp1',
    'thrombomodulin', 'epcr', 'fibrinogen', 'fibrin', 'fga', 'fgb', 'fgg',
    'plasminogen', 'plg', 'tpa', 'plasmin', 'p-selectin', 'selectin',
    'factor v leiden', 'prothrombin 20210', 'mthfr', 'jak2',
    'hmgb1', 'padi4', 'albumin', 'glucose', 'cholesterol', 'triglyceride',
}

MECHANISM_GENES = {
    'fut8', 'lgals3', 'galectin-3', 'cd44', 'itgb1', 'itga5', 'itgav',
    'mapk1', 'mapk3', 'rhoa', 'rock1', 'rock2', 'nfkb1', 'rela',
}

KNOWN_ANCHOR_MAP = {
    'rhoa': 'RHOA', 'rock1': 'ROCK1', 'rock2': 'ROCK2',
    'mapk1': 'MAPK1', 'mapk3': 'MAPK3', 'nfkb1': 'NFKB1', 'rela': 'RELA',
    'cd44': 'CD44', 'itgb1': 'ITGB1', 'itga5': 'ITGA5',
    'fut8': 'FUT8', 'lgals3': 'LGALS3',
}

PATHWAY_TERMS = {
    'Inflammation': ['inflamm', 'cytokine', 'il-', 'tnf', 'nf-kb', 'tlr', 'chemokine'],
    'Fibrosis': ['fibrosis', 'fibrotic', 'tgf', 'sma', 'collagen', 'mmp', 'timp', 'ctgf'],
    'Metabolism': ['metabol', 'lipid', 'glucose', 'insulin', 'cholesterol', 'fatty acid'],
    'Epigenetics': ['methyl', 'acetyl', 'hdac', 'sirt', 'mir', 'lncrna', 'histone'],
    'Oxidative Stress': ['oxid', 'ros', 'nadph', 'nox', 'sod', 'catalase', 'glutathione'],
    'Autophagy/Apoptosis': ['autophagy', 'apoptosis', 'caspase', 'bcl', 'bax', 'beclin'],
}

TARGET_EDGE_TYPES = [
    ("Gene", "ASSOCIATED_WITH", "Disease"),
    ("Gene", "PROMOTES", "Disease"),
    ("Gene", "INHIBITS", "Disease"),
    ("Gene", "CONTRIBUTES_TO", "Disease"),
    ("Gene", "PROTECTS", "Disease"),
    ("Protein", "ASSOCIATED_WITH", "Disease"),
    ("Protein", "CONTRIBUTES_TO", "Disease"),
]


def log(msg):
    print(msg, flush=True)


def _is_textbook(name: str) -> bool:
    n = name.lower()
    return any(term in n for term in TEXTBOOK_VTE_GENES)


def _is_mechanism_gene(name: str) -> bool:
    n = name.lower()
    return any(a in n for a in MECHANISM_GENES)


def _resolve_entity(name: str) -> str:
    """Map fragmented entity names to canonical gene symbols."""
    n = name.lower().strip()
    for fragment, canonical in ENTITY_RESOLUTION.items():
        if fragment in n:
            return canonical
    return name


def _resolve_anchor(name: str) -> str:
    n = name.lower()
    for k, v in KNOWN_ANCHOR_MAP.items():
        if k in n:
            return v
    return name[:30]


def _classify_pathways(name: str) -> list:
    n = name.lower()
    return [k for k, terms in PATHWAY_TERMS.items() if any(t in n for t in terms)]


def _build_train_adjacency(data, train_ei: dict) -> dict:
    adj = defaultdict(list)
    for et, ei in train_ei.items():
        src_t, rel, dst_t = et
        for j in range(ei.shape[1]):
            s, d = int(ei[0, j]), int(ei[1, j])
            adj[(src_t, s)].append((dst_t, d, rel))
            adj[(dst_t, d)].append((src_t, s, f"rev_{rel}"))
    return adj


def score_all_candidates_global(z_dict, src_type, dst_type, train_pairs,
                                 top_k=2000, device="cpu"):
    """Score ALL (src,dst) pairs via full matrix, mask train edges, global topk.

    No per-source truncation — every candidate pair is scored and ranked globally.
    Memory: O(num_src * num_dst) for one edge type at a time.
    """
    z_src = z_dict[src_type]  # [N_src, D]
    z_dst = z_dict[dst_type]  # [N_dst, D]
    num_src, num_dst = z_src.shape[0], z_dst.shape[0]

    # Build train-pair mask for this edge type
    # Use a boolean mask rather than setting individual -inf values
    existing = train_pairs

    # Process in chunks to manage memory for very large matrices
    chunk_size = 1000
    heap = []
    tiebreaker = 0

    n_total = 0
    for src_start in range(0, num_src, chunk_size):
        src_end = min(src_start + chunk_size, num_src)
        n_chunk = src_end - src_start
        scores_chunk = z_src[src_start:src_end] @ z_dst.T  # [chunk, N_dst]

        # Mask training pairs: set to very negative value
        for i_loc, src_idx in enumerate(range(src_start, src_end)):
            for dst_idx in existing.get(src_idx, []):
                if dst_idx < num_dst:
                    scores_chunk[i_loc, dst_idx] = -1e10

        # Get global top-K from this chunk
        k_per_chunk = min(top_k, scores_chunk.numel())
        top_vals, top_idx = torch.topk(scores_chunk.flatten(), k_per_chunk)
        top_rows = top_idx // num_dst  # local row in chunk
        top_cols = top_idx % num_dst

        for i in range(len(top_vals)):
            score = float(top_vals[i])
            if score <= -1e9:  # masked
                continue
            src_idx = src_start + int(top_rows[i].item())
            dst_idx = int(top_cols[i].item())
            entry = {
                "src_idx": src_idx,
                "dst_idx": dst_idx,
                "score": score,
            }
            if len(heap) < top_k:
                heapq.heappush(heap, (score, tiebreaker, entry))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, tiebreaker, entry))
            tiebreaker += 1
        n_total += scores_chunk.numel()

    # Extract and sort
    results = []
    while heap:
        s, _, e = heapq.heappop(heap)
        e["score"] = s
        results.append(e)
    results.reverse()  # highest first
    return results, n_total


def main():
    t0 = time.time()
    log("=" * 60)
    log(f"HIDDEN TARGET HUNTER v3 — Global Ranking + Entity Resolution")
    log(f"{datetime.now():%H:%M:%S}")
    log("=" * 60)

    # ── Config ──
    data_path = Path("data/processed/heterodata.pt")
    ckpt_dir = Path("checkpoints/full_training_v2")
    output_dir = Path("figures/hidden_targets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect best seed (or use specified seed)
    summary_path = ckpt_dir / "summary.json"
    best_seed_id = None
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        if "per_seed" in summary:
            best_seed = max(summary["per_seed"], key=lambda s: s["val_mrr"])
            best_seed_id = best_seed["seed"]
            log(f"  Best seed from summary: {best_seed_id}")
    if best_seed_id is None:
        seed_dirs = sorted(ckpt_dir.glob("seed_*"))
        if seed_dirs:
            best_seed_id = int(seed_dirs[0].name.split("_")[1])
        else:
            best_seed_id = 42  # default
        log(f"  Using seed_{best_seed_id} (auto-detected)")

    seed_dir = ckpt_dir / f"seed_{best_seed_id}"
    feat_cache = seed_dir / "features_cache.pt"
    ckpt_path = seed_dir / "checkpoint_best.pt"

    # ── 1. Load data + config ──
    log("\n[1/5] Loading KG data + config...")
    data = torch.load(data_path, weights_only=False)
    import yaml
    with open("config/anchor_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    config_ets = [tuple(et) for et in cfg.get("edge_types", [])]
    valid_nts = [nt for nt in data.node_types if data[nt].num_nodes > 0]
    meta_relations = [et for et in config_ets if et in data.edge_types]
    valid_ets = [(s, r, d) for s, r, d in meta_relations if s in valid_nts and d in valid_nts]
    log(f"  {len(valid_nts)} node types, {len(valid_ets)} edge types")

    # Load the EXACT train split used during training
    train_ei_path = seed_dir / "train_edges.pt"
    if train_ei_path.exists():
        train_ei = torch.load(train_ei_path, weights_only=False)
        log(f"  Loaded train split: {sum(ei.shape[1] for ei in train_ei.values())} edges")
    else:
        from data.temporal_split import RandomStratifiedSplitter
        splitter = RandomStratifiedSplitter(seed=best_seed_id, edge_types=valid_ets)
        train_ei, _, _ = splitter.split(data)
        train_ei = {et: ei for et, ei in train_ei.items() if et in valid_ets}
        log(f"  Reconstructed train split: {sum(ei.shape[1] for ei in train_ei.values())} edges")

    # ── 2. Load features + model ──
    log("\n[2/5] Loading features + model...")
    if feat_cache.exists():
        combined = torch.load(feat_cache, weights_only=False)
        feat_dim = combined[next(iter(combined))].shape[-1]
        for nt in valid_nts:
            data[nt].x = combined[nt]
        log(f"  Feature dim: {feat_dim}d")
    else:
        log("  ERROR: Feature cache not found. Run training first!")
        return

    # Load temperature config
    temp_init = {}
    for src, rel, dst in valid_ets:
        temp_init[f"{src}__{rel}__{dst}"] = 1.0
    for cat in cfg.get("temperature_init", {}).values():
        tau = cat.get("tau_init", 1.0)
        for rn in cat.get("relations", []):
            for src, rel, dst in valid_ets:
                if rel == rn:
                    temp_init[f"{src}__{rel}__{dst}"] = tau

    from models.tempered_hgt import TemperedHGT
    model = TemperedHGT(
        in_channels={nt: feat_dim for nt in valid_nts},
        hidden_channels=128, out_channels=128, num_heads=4, num_layers=2,
        meta_relations=valid_ets, temperature_init=temp_init,
    )

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        log(f"  Loaded epoch {ckpt['epoch']}, val_mrr={ckpt['val_mrr']:.4f}")
    else:
        log("  ERROR: No checkpoint found!")
        return
    model.eval()

    # ── 3. Message passing on TRAIN edges only ──
    log("\n[3/5] Message passing (train edges only)...")
    x_dict = {nt: data[nt].x for nt in valid_nts if nt in model.encoder.projections}
    with torch.no_grad():
        z_dict = model(x_dict, train_ei, cos_decay=0.0)

    # ── 4. Global candidate scoring ──
    log("\n[4/5] Scoring ALL candidate pairs globally (no per-source truncation)...")

    # Build train-pair lookup: {edge_type: {src_idx: set(dst_indices)}}
    train_pairs = defaultdict(lambda: defaultdict(set))
    for et, ei in train_ei.items():
        for j in range(ei.shape[1]):
            train_pairs[et][int(ei[0, j])].add(int(ei[1, j]))

    target_ets = [et for et in valid_ets
                  if et[2] == "Disease" and et[0] in ("Gene", "Protein")]
    TOP_K = 2000

    global_heap = []
    global_tb = 0
    total_scored = 0

    src_names = {}
    dst_names = {}
    for et in target_ets:
        src_t, rel, dst_t = et
        if src_t not in z_dict or dst_t not in z_dict:
            continue
        if src_t not in src_names:
            src_names[src_t] = data[src_t].name if hasattr(data[src_t], "name") else []
        if dst_t not in dst_names:
            dst_names[dst_t] = data[dst_t].name if hasattr(data[dst_t], "name") else []

        t_et = time.time()
        chunk_results, n_total = score_all_candidates_global(
            z_dict, src_t, dst_t,
            train_pairs.get(et, defaultdict(set)),
            top_k=TOP_K,
        )

        # Merge into global heap
        for entry in chunk_results:
            src_idx = entry["src_idx"]
            dst_idx = entry["dst_idx"]
            score = entry["score"]
            name = str(src_names[src_t][src_idx]) if src_idx < len(src_names[src_t]) else str(src_idx)
            disease = str(dst_names[dst_t][dst_idx]) if dst_idx < len(dst_names[dst_t]) else str(dst_idx)
            full_entry = {
                "name": name,
                "src_type": src_t,
                "relation": rel,
                "disease": disease,
                "score": score,
                "src_idx": src_idx,
                "dst_idx": dst_idx,
            }
            if len(global_heap) < TOP_K:
                heapq.heappush(global_heap, (score, global_tb, full_entry))
            elif score > global_heap[0][0]:
                heapq.heapreplace(global_heap, (score, global_tb, full_entry))
            global_tb += 1
        total_scored += n_total
        log(f"  {src_t}->{dst_t} ({rel}): {n_total:,} pairs scored in {time.time()-t_et:.1f}s")

    log(f"  Total pairs scored: {total_scored:,}")

    # ── 5. Entity resolution + ranking ──
    log("\n[5/5] Entity resolution + final ranking...")

    all_candidates = sorted(
        ({**e, "score": s} for s, _, e in global_heap),
        key=lambda p: -p["score"],
    )

    # Entity resolution: merge fragmented entities by canonical name
    resolved = {}  # canonical_name -> best score + metadata
    for c in all_candidates:
        canonical = _resolve_entity(c["name"])
        if canonical not in resolved or c["score"] > resolved[canonical]["score"]:
            resolved[canonical] = {
                "canonical_name": canonical,
                "original_name": c["name"],
                "src_type": c["src_type"],
                "disease": c["disease"],
                "score": c["score"],
                "relation": c["relation"],
                "is_par2": canonical == "F2RL1 (PAR-2)",
            }

    # Sort resolved entries by score
    ranked = sorted(resolved.values(), key=lambda p: -p["score"])

    # Print top 30
    print(f"\n{'='*80}")
    print(f"TOP-30 CANDIDATE TARGETS (global ranking, entity-resolved)")
    print(f"{'='*80}")
    for i, c in enumerate(ranked[:30]):
        par2_mark = " ★ PAR-2" if c["is_par2"] else ""
        print(f"{i+1:3d}. {c['canonical_name'][:50]:50s} "
              f"[{c['src_type']:7s}] score={c['score']:.4f}  {c['disease'][:25]}{par2_mark}")

    # PAR-2 specific report
    print(f"\n{'='*80}")
    print(f"PAR-2 (F2RL1) ENTITY RESOLUTION REPORT")
    print(f"{'='*80}")
    par2_all = [c for c in ranked if c["is_par2"]]
    if par2_all:
        for c in par2_all:
            idx = ranked.index(c) + 1
            print(f"  Canonical rank: {idx}/{len(ranked)}")
            print(f"  Original name:  {c['original_name']}")
            print(f"  Best disease:   {c['disease']}")
            print(f"  Score:          {c['score']:.4f}")
    else:
        print(f"  PAR-2 NOT FOUND in top {len(ranked)} entity-resolved candidates")
        print(f"  Checking raw (pre-resolution) candidates...")
        for c in all_candidates:
            n = c["name"].lower()
            if any(k in n for k in ["par-2", "par2", "f2rl1"]):
                idx = all_candidates.index(c) + 1
                print(f"  Raw rank {idx}: {c['name']} score={c['score']:.4f} disease={c['disease']}")

    # Save
    export = []
    for i, c in enumerate(ranked):
        export.append({**c, "rank": i + 1})
    with open(output_dir / "full_ranked_candidates.json", "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output_dir / 'full_ranked_candidates.json'}")
    print(f"Time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
