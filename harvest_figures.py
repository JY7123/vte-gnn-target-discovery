#!/usr/bin/env python3
"""Figure 3/4 data harvesting pipeline.

Loads trained TemperedHGT, runs forward pass on test set, extracts Top-50
novel predictions, applies 2-tier filtering (drug repurposing / novel pathway),
runs GNNExplainer + literature validation + MR cross-check, outputs
Figure-ready JSON/CSV.

Tier 1: Drug repurposing — predicted Drug→Disease edges with existing FDA drugs
Tier 2: Novel pathway — predicted Gene→Disease edges mechanistically linked
         to FUT8/Lgals3/CD44 or Focal adhesion axes, unseen in training

Usage:
    python harvest_figures.py --model checkpoints/best.pt --output figures/
"""
import argparse
import json
import csv
import sys
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import torch

def parse_args():
    p = argparse.ArgumentParser(description="Harvest Figure 3/4 data from trained TemperedHGT")
    p.add_argument("--model", default=None, help="Path to model checkpoint (optional — builds from scratch if missing)")
    p.add_argument("--data", default="data/processed", help="Path to Phase 1 processed data")
    p.add_argument("--output", default="figures/", help="Output directory for Figure data")
    p.add_argument("--top-k", type=int, default=50, help="Number of top predictions to extract")
    p.add_argument("--tier1-k", type=int, default=10, help="Top-N for drug repurposing tier")
    p.add_argument("--tier2-k", type=int, default=10, help="Top-N for novel pathway tier")
    p.add_argument("--hidden", type=int, default=128, help="Hidden channels for model")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_data(data_dir):
    """Load Phase 1 processed HeteroData and edge splits."""
    data_dir = Path(data_dir)
    data = torch.load(data_dir / "heterodata.pt", weights_only=False)
    train_ei = torch.load(data_dir / "train_edges.pt", weights_only=False)
    test_ei = torch.load(data_dir / "test_edges.pt", weights_only=False)
    neg_ei = torch.load(data_dir / "negative_edges.pt", weights_only=False)
    return data, train_ei, test_ei, neg_ei


def build_model(data, args):
    """Build or load TemperedHGT model."""
    from models.tempered_hgt import TemperedHGT

    meta_relations = list(data.edge_types)
    node_types = list(data.node_types)

    # Build input channels — auto-detect from checkpoint if available
    in_channels = {}
    ckpt_in_dim = None
    if args.model and Path(args.model).exists():
        ckpt = torch.load(args.model, weights_only=True)
        for key, val in ckpt["model_state_dict"].items():
            if "encoder.projections." in key and key.endswith(".weight"):
                ckpt_in_dim = val.shape[1]
                break
    in_dim = ckpt_in_dim or 896
    for nt in node_types:
        in_channels[nt] = in_dim

    # Build temperature init from config
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

    # Default tau for any remaining
    for src, rel, dst in meta_relations:
        key = f"{src}__{rel}__{dst}"
        if key not in temp_init:
            temp_init[key] = 1.0

    model = TemperedHGT(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        out_channels=args.hidden,
        num_heads=4,
        num_layers=2,
        meta_relations=meta_relations,
        temperature_init=temp_init,
    )

    if args.model and Path(args.model).exists() and ckpt_in_dim is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model from {args.model} (epoch {ckpt.get('epoch', '?')})")

    model.to(args.device)
    model.eval()
    return model, meta_relations, node_types, in_dim


def predict_all_edges(model, data, train_ei, args, in_dim=None):
    """Forward pass: score all possible Gene→Disease and Drug→Disease edges."""
    print("\n[1/6] Running forward pass for link prediction...")

    # Detect input dimension from model encoder
    if in_dim is None:
        in_dim = 896

    # Build feature dict
    x_dict = {}
    for nt in data.node_types:
        if hasattr(data[nt], 'x') and data[nt].x is not None:
            x_dict[nt] = data[nt].x.to(args.device)
        else:
            x_dict[nt] = torch.randn(data[nt].num_nodes, in_dim, device=args.device)

    edge_index_dict = {et: data[et].edge_index.to(args.device) for et in data.edge_types}

    with torch.no_grad():
        z = model(x_dict, edge_index_dict, cos_decay=0.0)

    # Collect training edges for filtering
    train_edge_set = set()
    for et, ei in train_ei.items():
        for i in range(ei.shape[1]):
            train_edge_set.add((et, int(ei[0, i]), int(ei[1, i])))

    # Score target edge types: Gene→Disease, Drug→Disease, Drug→Protein
    target_edge_types = []
    for et in data.edge_types:
        src, rel, dst = et
        if dst in ("Disease",) and src in ("Gene", "Drug", "Protein", "Cytokine"):
            target_edge_types.append(et)

    predictions = []
    for et in target_edge_types:
        src_t, rel, dst_t = et
        if et not in data.edge_types:
            continue

        # Score edges NOT in training set
        num_src = data[src_t].num_nodes
        num_dst = data[dst_t].num_nodes

        # For efficiency, score random subset if too many candidates
        max_candidates = min(num_src * num_dst, 50000)
        if num_src * num_dst > max_candidates:
            src_sample = torch.randint(0, num_src, (max_candidates,))
            dst_sample = torch.randint(0, num_dst, (max_candidates,))
        else:
            src_idx = torch.arange(num_src).repeat_interleave(num_dst)
            dst_idx = torch.arange(num_dst).repeat(num_src)
            src_sample = src_idx[:max_candidates]
            dst_sample = dst_idx[:max_candidates]

        candidates = torch.stack([src_sample, dst_sample])

        # Filter out training edges
        novel_mask = []
        for i in range(candidates.shape[1]):
            s, d = int(candidates[0, i]), int(candidates[1, i])
            novel_mask.append((et, s, d) not in train_edge_set)
        novel_mask = torch.tensor(novel_mask)
        novel_candidates = candidates[:, novel_mask]

        if novel_candidates.shape[1] == 0:
            continue

        with torch.no_grad():
            scores = model.decode(z, novel_candidates.to(args.device), src_t, dst_t)

        # Map indices back to names
        src_names = data[src_t].name if hasattr(data[src_t], 'name') else [str(i) for i in range(num_src)]
        dst_names = data[dst_t].name if hasattr(data[dst_t], 'name') else [str(i) for i in range(num_dst)]

        for i in range(len(scores)):
            s_idx = int(novel_candidates[0, i])
            d_idx = int(novel_candidates[1, i])
            predictions.append({
                "edge_type": list(et),
                "src_type": src_t,
                "dst_type": dst_t,
                "relation": rel,
                "src_name": src_names[s_idx] if s_idx < len(src_names) else str(s_idx),
                "dst_name": dst_names[d_idx] if d_idx < len(dst_names) else str(d_idx),
                "src_idx": s_idx,
                "dst_idx": d_idx,
                "score": float(scores[i]),
            })

    # Sort by score descending
    predictions.sort(key=lambda x: x["score"], reverse=True)
    print(f"  Scored {len(predictions)} novel edges across {len(target_edge_types)} edge types")
    return predictions[:args.top_k], z, x_dict


def apply_tier_filter(predictions, data, args):
    """Apply 2-tier filtering: drug repurposing (Tier 1) + novel pathway (Tier 2).

    Tier 1: Drug→Disease or Drug→Protein edges with high confidence
    Tier 2: Gene→Disease edges linked to FUT8/Lgals3/CD44/Focal adhesion pathways
    """
    print("\n[2/6] Applying tier filters...")

    # Core pathway genes (FUT8 axis + Focal adhesion)
    core_pathway = {
        "FUT8", "Lgals3", "CD44", "ITGB1", "ITGA5", "ITGAV",
        "TLN1", "VCL", "ACTN1", "ACTB", "RHOA", "ROCK1", "ROCK2",
        "MAPK1", "MAPK3", "NFKB1", "RELA", "TNF", "TLR4",
        "F2", "F11", "KNG1", "LRP4",
    }

    tier1 = []  # Drug repurposing
    tier2 = []  # Novel pathway
    tier3 = []  # Other high-confidence

    for pred in predictions:
        src_name = pred["src_name"].lower()
        src_type = pred["src_type"]

        if src_type == "Drug":
            tier1.append(pred)
        elif any(gene.lower() in src_name for gene in core_pathway):
            tier2.append(pred)
        elif pred["score"] > 0.8:
            tier3.append(pred)

    print(f"  Tier 1 (drug repurposing): {len(tier1)} candidates")
    print(f"  Tier 2 (novel pathway):    {len(tier2)} candidates")
    print(f"  Tier 3 (other high-conf):  {len(tier3)} candidates")

    return {
        "tier1_drug_repurposing": tier1[:args.tier1_k],
        "tier2_novel_pathway": tier2[:args.tier2_k],
        "tier3_other": tier3[:args.tier2_k],
        "all_top": predictions[:args.top_k],
    }


def run_explainability(tiered, model, data, x_dict, args):
    """Run GNNExplainer on top predictions per tier."""
    print("\n[3/6] Running GNNExplainer on top targets...")

    from explainability.gnnexplainer_vte import VTEExplainer
    from explainability.alignment_engine import AnchorAlignmentEngine
    from explainability.contradiction_gate import ContradictionGate

    explainer = VTEExplainer(model, num_epochs=100)

    # Load anchors from config
    cfg_path = Path("config/anchor_config.yaml")
    anchors = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        gnn_cfg = cfg.get("gnnexplainer_anchors", {})
        anchors["positive"] = gnn_cfg.get("positive_anchors", [])
        anchors["pathway"] = gnn_cfg.get("pathway_anchors", [])
        anchors["negative"] = gnn_cfg.get("negative_anchors", [])

    alignment = AnchorAlignmentEngine(
        positive_anchors=anchors.get("positive", []),
        pathway_anchors=anchors.get("pathway", []),
        negative_anchors=anchors.get("negative", []),
    )
    contradiction = ContradictionGate(
        falsified_targets=anchors.get("negative", ["Padi4", "Hmgb1"])
    )

    # Build gene name lookup
    gene_names = {}
    for nt in data.node_types:
        if hasattr(data[nt], 'name'):
            for idx, name in enumerate(data[nt].name):
                gene_names[idx] = name

    results = {"tier1": [], "tier2": [], "alignment": {}, "contradiction": {}}

    # Explain top-3 per tier
    for tier_name in ["tier1_drug_repurposing", "tier2_novel_pathway"]:
        tier_preds = tiered.get(tier_name, [])[:3]

        for pred in tier_preds:
            et = tuple(pred["edge_type"])
            if et not in data.edge_types:
                continue

            # Get edge index for this type to find matching edge
            ei = data[et].edge_index
            mask = (ei[0] == pred["src_idx"]) & (ei[1] == pred["dst_idx"])

            if mask.any():
                edge_idx = mask.nonzero(as_tuple=True)[0][0].item()
            else:
                continue

            # Run GNNExplainer
            try:
                explanation = explainer.explain_edge(data, et, edge_idx, x_dict)
                explanation["prediction_score"] = pred["score"]
                explanation["src_name"] = pred["src_name"]
                explanation["dst_name"] = pred["dst_name"]
                explanation["relation"] = pred["relation"]

                # Build explained genes set for alignment
                explained_genes = set()
                # Add source and destination
                explained_genes.add(pred["src_name"])
                explained_genes.add(pred["dst_name"])

                # Classify
                align_result = alignment.classify_target(explained_genes, pred["score"])
                explanation["alignment"] = align_result

                # Contradiction check
                # Build minimal path edges from explanation
                expl_edges = {}
                for expl_et, mask_vals in explanation.get("edge_mask", {}).items():
                    if isinstance(mask_vals, list):
                        expl_edges[expl_et] = [(int(data[expl_et].edge_index[0, j]),
                                                  int(data[expl_et].edge_index[1, j]))
                                                 for j in range(min(len(mask_vals),
                                                                    data[expl_et].edge_index.shape[1]))
                                                 if float(mask_vals[j]) > 0.1]
                contradiction_result = contradiction.check_path(expl_edges, gene_names)
                explanation["contradiction"] = contradiction_result

                tier_key = "tier1" if "tier1" in tier_name else "tier2"
                results[tier_key].append(explanation)

            except Exception as e:
                print(f"  Warning: GNNExplainer failed for {pred['src_name']}->{pred['dst_name']}: {e}")

    # Radar data for all explained targets
    radar_targets = {}
    for exp in results["tier1"] + results["tier2"]:
        name = f"{exp.get('src_name', '?')}->{exp.get('dst_name', '?')}"
        explained = set()
        explained.add(exp.get("src_name", ""))
        explained.add(exp.get("dst_name", ""))
        radar_targets[name] = explained

    results["alignment"] = alignment.build_radar_data(radar_targets)

    print(f"  Explained Tier 1: {len(results['tier1'])} targets")
    print(f"  Explained Tier 2: {len(results['tier2'])} targets")

    return results


def run_literature_validation(tiered, args):
    """Run temporal-locked literature backcheck."""
    print("\n[4/6] Running literature validation...")

    from validation.literature_validation import LiteratureValidator
    validator = LiteratureValidator()

    all_predictions = tiered["tier1_drug_repurposing"] + tiered["tier2_novel_pathway"]

    # Build list of unique gene/drug names for PubMed search
    unique_targets = list(set(p["src_name"] for p in all_predictions))

    lit_results = validator.validate_predictions([
        {"gene": t, "score": 0.0} for t in unique_targets
    ])

    # Enrich tiered predictions with literature status
    for pred in all_predictions:
        pred["literature_status"] = "unvalidated"

    lit_results["target_count"] = len(unique_targets)
    lit_results["timestamp"] = datetime.now().isoformat()

    print(f"  Validated {len(unique_targets)} unique targets")
    return lit_results


def run_mr_crosscheck(tiered, args):
    """Run MR causal cross-validation."""
    print("\n[5/6] Running MR cross-validation...")

    from validation.cross_check_mr import MRCrossValidator
    validator = MRCrossValidator()

    all_predictions = tiered["tier1_drug_repurposing"] + tiered["tier2_novel_pathway"]

    # Map predictions to gene-centric format
    gnn_predictions = []
    seen = set()
    for pred in all_predictions:
        gene = pred["src_name"]
        if gene not in seen:
            gnn_predictions.append({"gene": gene, "score": pred["score"]})
            seen.add(gene)

    overlap = validator.compute_overlap(gnn_predictions)
    venn = validator.build_venn_data(gnn_predictions)

    # Enrich with MR info
    for pred in all_predictions:
        pred["is_mr_target"] = validator.is_mr_target(pred["src_name"])
        if pred["is_mr_target"]:
            pred["mr_info"] = validator.get_mr_info(pred["src_name"])

    print(f"  MR overlap: {overlap['intersection']}/{len(validator.mr_gene_set)} MR targets hit")
    print(f"  Intersection genes: {overlap['intersection_genes']}")

    return {**overlap, "venn_data": venn}


def export_figures(tiered, explanations, lit_results, mr_results, args):
    """Export all Figure data as JSON/CSV for rendering."""
    print("\n[6/6] Exporting Figure data...")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Figure 3A/B: Top predictions bar chart data
    fig3_predictions = []
    for pred in tiered["all_top"]:
        fig3_predictions.append({
            "target": pred["src_name"],
            "target_type": pred["src_type"],
            "disease": pred["dst_name"],
            "relation": pred["relation"],
            "score": pred["score"],
            "tier": ("drug_repurposing" if pred["src_type"] == "Drug"
                     else "novel_pathway" if pred.get("is_mr_target")
                     else "other"),
            "is_mr_target": pred.get("is_mr_target", False),
        })

    with open(out / "fig3_predictions.json", "w") as f:
        json.dump(fig3_predictions, f, indent=2)

    # CSV version for easy Excel import
    with open(out / "fig3_predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "target_type", "disease", "relation", "score", "tier", "is_mr_target"])
        writer.writeheader()
        writer.writerows(fig3_predictions)

    # Figure 3C: Literature validation table
    with open(out / "fig3_literature_validation.json", "w") as f:
        json.dump(lit_results, f, indent=2, default=str)

    # Figure 3D: MR Venn data
    with open(out / "fig3_mr_venn.json", "w") as f:
        json.dump(mr_results, f, indent=2)

    # Figure 4A: Subgraph data for Cytoscape
    from explainability.subgraph_extractor import SubgraphExtractor
    extractor = SubgraphExtractor()

    for exp in explanations["tier1"] + explanations["tier2"]:
        name = f"{exp.get('src_name', 'unknown')}__{exp.get('relation', 'REL')}__{exp.get('dst_name', 'unknown')}"
        name = name.replace("/", "_").replace(" ", "_")

        # Build edge_src_dst from data
        edge_src_dst = {}
        for et, mask_vals in exp.get("edge_mask", {}).items():
            if et in exp.get("_data_edge_types", {}):
                edge_src_dst[et] = exp["_data_edge_types"].get(et, [])

        try:
            extractor.to_json(exp, edge_src_dst,
                              str(out / f"fig4a_subgraph_{name}.json"),
                              threshold=0.1)
        except Exception:
            pass

    # Figure 4B: Radar data
    with open(out / "fig4b_radar.json", "w") as f:
        json.dump(explanations.get("alignment", {}), f, indent=2)

    # Figure 4C: Contradiction heatmap data
    contradiction_data = []
    for exp in explanations["tier1"] + explanations["tier2"]:
        contradiction_data.append({
            "target": f"{exp.get('src_name', '?')}->{exp.get('dst_name', '?')}",
            "contradiction_score": exp.get("contradiction", {}).get("contradiction_score", 0),
            "contaminated": exp.get("contradiction", {}).get("contaminated", False),
            "contaminated_nodes": exp.get("contradiction", {}).get("contaminated_nodes", []),
            "prediction_score": exp.get("prediction_score", 0),
        })

    with open(out / "fig4c_contradiction.json", "w") as f:
        json.dump(contradiction_data, f, indent=2)

    # Master summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_predictions_scored": len(tiered["all_top"]),
        "tier1_drug_repurposing": len(tiered["tier1_drug_repurposing"]),
        "tier2_novel_pathway": len(tiered["tier2_novel_pathway"]),
        "mr_intersection_genes": mr_results.get("intersection_genes", []),
        "top_targets": [
            {"rank": i+1, "target": p["src_name"], "score": p["score"],
             "tier": "drug" if p["src_type"] == "Drug" else "pathway" if p.get("is_mr_target") else "other"}
            for i, p in enumerate(tiered["all_top"][:10])
        ],
        "output_files": [str(p.relative_to(out)) for p in sorted(out.glob("*"))],
    }

    with open(out / "harvest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Figure data exported to {out.absolute()}")
    print(f"Files: {len(list(out.glob('*')))} generated")
    print(f"Top predicted target: {tiered['all_top'][0]['src_name']} "
          f"→ {tiered['all_top'][0]['dst_name']} "
          f"(score: {tiered['all_top'][0]['score']:.4f})")
    print(f"{'='*60}")

    return summary


def main():
    args = parse_args()
    print("=" * 60)
    print("VTE GNN Figure 3/4 Data Harvesting Pipeline")
    print("=" * 60)

    # Load data
    data, train_ei, test_ei, neg_ei = load_data(args.data)
    print(f"Loaded: {sum(data[nt].num_nodes for nt in data.node_types)} nodes, "
          f"{sum(data[et].edge_index.shape[1] for et in data.edge_types)} edges")

    # Build/load model
    model, meta_relations, node_types, in_dim = build_model(data, args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, {len(meta_relations)} edge types")

    # Step 1: Score all novel edges
    top_predictions, z, x_dict = predict_all_edges(model, data, train_ei, args, in_dim=in_dim)

    # Step 2: Apply tier filters
    tiered = apply_tier_filter(top_predictions, data, args)

    # Step 3: GNNExplainer + Alignment + Contradiction
    explanations = run_explainability(tiered, model, data, x_dict, args)

    # Step 4: Literature validation
    lit_results = run_literature_validation(tiered, args)

    # Step 5: MR cross-check
    mr_results = run_mr_crosscheck(tiered, args)

    # Step 6: Export
    summary = export_figures(tiered, explanations, lit_results, mr_results, args)

    return summary


if __name__ == "__main__":
    main()
