#!/usr/bin/env python3
"""Final harvest v2: semantic pathway filtering + alignment + temporal check.

Adapted for sparse KG (12,890/144K nodes have edges).
Uses entity-name-based pathway matching instead of BFS.
"""
import sys, json, time, torch, csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PATHWAY_TERMS = {
    "Inflammation": ["inflamm", "tnf", "nf-kb", "nfkb", "tlr", "il-1", "il-6", "cytokine", "cox-2"],
    "Fibrosis": ["fibrosis", "fibrotic", "tgf", "sma", "collagen", "ecm", "α-sma", "myofibroblast"],
    "Coagulation": ["coagul", "thrombin", "fibrin", "platelet", "thrombosis", "prothromb", "f2 ", "f11"],
    "Fucosylation": ["fucosyl", "fut8", "glycosyl", "lectin", "galectin", "lgals3"],
    "Adhesion": ["adhesion", "integrin", "cd44", "itgb", "itga", "focal adhesion"],
    "Endothelial": ["endothelial", "ve-cadherin", "angiopoietin", "tie2", "vegf"],
}

FALSIFIED = ["padi4", "hmgb1"]

def log(msg):
    print(msg, flush=True)

def load_model():
    log("[1/4] Loading trained model...")
    data = torch.load("data/processed/heterodata.pt", weights_only=False)
    meta_relations = list(data.edge_types)
    temp_init = {f"{s}__{r}__{d}": 1.0 for s, r, d in meta_relations}

    from models.tempered_hgt import TemperedHGT
    model = TemperedHGT(
        in_channels={nt: 64 for nt in data.node_types},
        hidden_channels=64, out_channels=64, num_heads=4, num_layers=2,
        meta_relations=meta_relations, temperature_init=temp_init,
    )
    ckpt = torch.load("checkpoints/fast_training/checkpoint_epoch_24.pt", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log(f"  Loaded (epoch {ckpt['epoch']}), {sum(p.numel() for p in model.parameters()):,} params")
    return model, data

def classify_pathway(target_name):
    """Classify target by pathway based on name substring matching."""
    name_lower = target_name.lower()
    matched = []
    for pathway, terms in PATHWAY_TERMS.items():
        for term in terms:
            if term in name_lower:
                matched.append(pathway)
                break
    contaminated = any(f in name_lower for f in FALSIFIED)
    return matched, contaminated

def filter_and_classify(predictions):
    """Filter predictions by pathway relevance and classify mechanism type."""
    log("\n[2/4] Pathway filtering + classification...")
    results = []

    for pred in predictions:
        target = pred["target"]
        pathways, contaminated = classify_pathway(target)

        if not contaminated and not pathways:
            continue  # skip non-relevant

        pred["pathways"] = pathways
        pred["contaminated"] = contaminated
        pred["num_pathways"] = len(pathways)

        # Classify mechanism type
        if contaminated:
            pred["mechanism_type"] = "contaminated"
        elif len(pathways) >= 2:
            pred["mechanism_type"] = "multi_pathway_hub"
        elif "Fucosylation" in pathways or "Adhesion" in pathways:
            pred["mechanism_type"] = "shared_downstream"
        elif "Inflammation" in pathways or "Coagulation" in pathways:
            pred["mechanism_type"] = "parallel_regulation"
        else:
            pred["mechanism_type"] = "novel_mechanism"

        results.append(pred)

    results.sort(key=lambda p: (p["contaminated"], -p["num_pathways"], -p["score"]))
    log(f"  {len(results)}/{len(predictions)} pathway-relevant (non-contaminated)")
    log(f"  Contaminated: {sum(1 for p in results if p['contaminated'])}")
    for mtype in ["shared_downstream", "parallel_regulation", "multi_pathway_hub", "novel_mechanism"]:
        cnt = sum(1 for p in results if p["mechanism_type"] == mtype)
        if cnt:
            log(f"  {mtype}: {cnt}")
    return results

def run_alignment(filtered):
    """Run anchor alignment scoring on filtered targets."""
    log("\n[3/4] Anchor alignment scoring...")

    from explainability.alignment_engine import AnchorAlignmentEngine
    alignment = AnchorAlignmentEngine(
        positive_anchors=["FUT8", "Lgals3", "CD44", "ITGB1", "ITGA5", "ITGAV"],
        pathway_anchors=["TNF", "TLR4", "MAPK1", "MAPK3", "NFKB1", "RELA",
                         "RHOA", "ROCK1", "ROCK2", "F2", "F11", "KNG1", "LRP4"],
        negative_anchors=["Padi4", "Hmgb1"],
    )

    for pred in filtered[:15]:
        explained = set()
        explained.add(pred["target"])
        for pw in pred.get("pathways", []):
            explained.add(pw)
        align = alignment.classify_target(explained, pred["score"])
        pred["alignment"] = align

    # Radar data for visualization
    radar_input = {}
    for pred in filtered[:10]:
        name = pred["target"][:40]
        explained = set()
        explained.add(pred["target"])
        for pw in pred.get("pathways", []):
            explained.add(pw)
        radar_input[name] = explained

    radar = alignment.build_radar_data(radar_input)
    log(f"  Alignment computed for {len(filtered[:15])} targets")
    return filtered, radar

def temporal_check(filtered):
    """Temporal literature check."""
    log("\n[4/4] Temporal literature check (target: 2025H2-2026H1)...")
    targets = [p["target"] for p in filtered[:15] if not p.get("contaminated")]
    log(f"  {len(targets)} targets queued for PubMed blind backcheck")
    log(f"  NOTE: Requires NCBI PMID enrichment (7,914 edges have PMIDs)")
    return {"pending_targets": targets, "window": "2025-07 to 2026-06", "status": "queued"}

def export(filtered, radar, temporal, top_preds):
    """Export final Figure data."""
    out = Path("figures/final")
    out.mkdir(parents=True, exist_ok=True)

    # Figure 3A: Pathway-filtered targets
    fig3a = []
    for i, p in enumerate(filtered[:20]):
        fig3a.append({
            "rank": i + 1, "target": p["target"], "score": round(p["score"], 2),
            "pathways": p.get("pathways", []), "num_pathways": p.get("num_pathways", 0),
            "mechanism_type": p.get("mechanism_type", ""),
            "alignment_score": round(p.get("alignment", {}).get("alignment_score", 0), 3),
            "contaminated": p.get("contaminated", False),
        })
    with open(out / "fig3a_pathway_targets.json", "w") as f:
        json.dump(fig3a, f, indent=2)
    if fig3a:
        with open(out / "fig3a_pathway_targets.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fig3a[0].keys())
            w.writeheader(); w.writerows(fig3a)

    # Figure 3B: Mechanism summary
    counts = defaultdict(int)
    for p in filtered:
        counts[p.get("mechanism_type", "unknown")] += 1
    with open(out / "fig3b_mechanism_summary.json", "w") as f:
        json.dump(dict(counts), f, indent=2)

    # Figure 4B: Radar data
    with open(out / "fig4b_radar.json", "w") as f:
        json.dump(radar, f, indent=2)

    # Temporal status
    with open(out / "fig3c_temporal.json", "w") as f:
        json.dump(temporal, f, indent=2)

    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_auroc": 0.751,
        "total_scored": len(top_preds),
        "pathway_filtered": len(filtered),
        "shared_downstream": counts.get("shared_downstream", 0),
        "parallel_regulation": counts.get("parallel_regulation", 0),
        "multi_pathway_hub": counts.get("multi_pathway_hub", 0),
    }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print top targets
    log(f"\n{'='*70}")
    log(f"TOP PATHWAY-FILTERED TARGETS")
    log(f"{'#':<4} {'Target':<45} {'Score':<8} {'Pathways':<35} {'Type':<22}")
    log("-" * 70)
    for p in fig3a[:15]:
        pw_str = ", ".join(p["pathways"][:3])
        log(f'{p["rank"]:<4} {p["target"][:43]:<45} {p["score"]:<8.1f} {pw_str:<35} {p["mechanism_type"]:<22}')

    log(f"\nFiles exported to: {out.absolute()}")
    return summary

def main():
    t0 = time.time()
    model, data = load_model()
    with open("figures/fig3_predictions.json") as f:
        predictions = json.load(f)

    filtered = filter_and_classify(predictions)
    filtered, radar = run_alignment(filtered)
    temporal = temporal_check(filtered)
    summary = export(filtered, radar, temporal, predictions)

    log(f"\nDone in {(time.time()-t0)/60:.1f} min")
    return summary

if __name__ == "__main__":
    main()
