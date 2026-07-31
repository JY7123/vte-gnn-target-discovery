# GNN Manuscript Revision Design

## Context

Nature Communications reviewer audited code, identified data leakage + anchor forcing + non-standard metrics. All code fixed. New training results (5 seeds × 100 epochs): Test AUROC 0.7409 ± 0.0749, Filtered MRR 0.0857 ± 0.0288. PAR-2 not in top 133 candidates. Old narrative ("GNN discovers PAR-2") is unsalvageable.

## New Narrative

**"Graph learning prioritizes cell-type-specific inflammatory and fibrotic programs in VTE"**

GNN's value is not guessing a single target, but systematically prioritizing two core pathological network axes from the literature knowledge graph:
- **Axis A (Macrophage Inflammation)**: TLR4 / NFKB1 / TGFB1 / SPP1
- **Axis B (Fibroblast Fibrosis)**: SMAD4 / COL1A1 / FN1 / ACTA2

scRNA-seq independently maps these two axes to specific cellular niches in the thrombosed vein wall.

## Proposed Title

*Temporally Evaluated Heterogeneous Graph Learning Prioritizes Cell-Type-Specific Inflammatory and Fibrotic Programs in Venous Thromboembolism*

Target journal: Cell Reports, ATVB, or Briefings in Bioinformatics

## Figures (5 Main Figures)

### Figure 1: Tempered HGT Architecture + Temporal Split Framework
- Panel A: Heterogeneous KG schema (14 node types, 5,056 → 29 curated edge types)
- Panel B: Tempered HGT architecture (PubMedBERT 768d + Node2Vec 128d → 896d, 4-head 2-layer, relation-specific temperature τ)
- Panel C: Temporal split design (train ≤ 2024, val/test 2025-2026; fallback: random stratified 80/10/10)
- Panel D: Anti-leakage evaluation schematic (message passing on train only, filtered ranking)

### Figure 2: Benchmark Performance + Ablation
- Panel A: 5-seed test metrics bar chart (AUROC, MRR, Hits@10) with error bars
- Panel B: Baseline comparison table (TransE, DistMult, ComplEx, RotatE, RGCN vs TemperedHGT)
- Panel C: Per-relation filtered MRR breakdown
- Panel D: Ablation study (temperature on/off, PubMedBERT/Node2Vec feature components)

### Figure 3: GNN Global Prioritization of VTE Pathological Programs
- Panel A: Top-30 ranked targets bar chart (entity-resolved, colored by pathway annotation)
- Panel B: Network visualization: two-axis layout (Inflammation Hub + Fibrosis Hub) with target connections
- Panel C: Comparison with traditional literature frequency ranking (PubMed count vs GNN score)
- Panel D: Degree vs GNN score scatter plot (showing GNN rewards more than just high-degree nodes)

### Figure 4: scRNA-seq Cell-Type Mapping of GNN-Prioritized Network Genes
Data source: Existing AnnData (21,230 cells, 8 cell types, IVC stenosis model Day 14)
- Panel A: UMAP overview (cell type annotation, DVT vs Control split)
- Panel B: Dotplot/bubble heatmap — GNN-prioritized genes × cell types (TLR4, NFKB1, TGFB1, SPP1 → Macrophage; SMAD4, COL1A1, FN1 → Fibroblast)
- Panel C: TGFB1 (ligand, macrophage) → SMAD4/COL1A1 (target, fibroblast) expression violin plots split by condition
- Panel D: CellChat macrophage→fibroblast TGF-β signaling pathway (putative)

### Figure 5: Cross-Species Validation in Human VTE Cohorts
- Panel A: GSEA enrichment of GNN-prioritized gene sets (Inflammation Program + Fibrosis Program) in GSE48000 human VTE whole blood
- Panel B: Leave-one-gene-out robustness analysis
- Panel C: Comparison with random gene sets (size-matched negative control)

## Implementation Tasks

### Phase 1: Run Baselines (required for Figure 2)
- TransE, DistMult, ComplEx, RotatE on same train/val/test split
- Filtered MRR/Hits@K comparison
- Time: ~2-3 hours CPU

### Phase 2: Figure Generation Scripts
- `render_figure3_target_ranking.py` — top-30 bar chart + network visualization + degree scatter
- `render_figure4_scRNA_mapping.py` — dotplot + violin + CellChat integration from AnnData
- `render_figure5_cross_species.py` — GSEA from GSE48000
- Update existing `render_figure2.R` for AUROC/MRR/Hits@10 with error bars

### Phase 3: Manuscript Rewrite
- Abstract: drop PAR-2, add "cell-type-specific programs"
- Introduction: reframe from "target discovery" to "systematic prioritization"
- Results Section 3: GNN global ranking (Figure 3)
- Results Section 4: scRNA cellular niche mapping (Figure 4)
- Results Section 5: Human cross-species validation (Figure 5)
- Discussion: limitations (no in vivo validation, temporal split limited by PubMed date availability)

### Phase 4: Reviewer Response Letter
- Point-by-point response to all 12 GNN manuscript critiques
- Document all code fixes with commit hashes
- Explain narrative shift from "discovery" to "prioritization"

## Key Messages to Convey

1. **Honesty**: GNN correctly prioritizes high-evidence targets; we don't claim it "discovered" them
2. **Complementarity**: GNN provides macro-level topology; scRNA-seq provides micro-level cell-type resolution
3. **Rigor**: No data leakage, filtered metrics, 5-seed statistics, entity resolution
4. **Translation**: Cross-species validation in human cohorts
