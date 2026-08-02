# Supplementary Materials

**Manuscript:** "Data-Leakage-Free Heterogeneous Graph Learning Prioritizes Cell-Type-Specific Inflammatory and Fibrotic Programs in Venous Thromboembolism"

---

## Table S1: Data Lineage and Version Tracking

| Resource | Version / Identifier | Description | Source |
|----------|---------------------|-------------|--------|
| Knowledge Graph | v2.1 (Zenodo DOI: 10.5281/zenodo.21757258) | 82,644 nodes, 14 entity types, 29 curated edge types, 11,989 edges | Built from PubMed abstracts + PMC full-text + curated databases |
| Training split | train_edges.pt (per-seed) | 9,591 edges (80%), random stratified | `data/processed/heterodata.pt` |
| Validation split | val_edges.pt (per-seed) | 1,199 edges (10%), random stratified | `data/processed/heterodata.pt` |
| Test split | test_edges.pt (per-seed) | 1,199 edges (10%), random stratified | `data/processed/heterodata.pt` |
| Node features (PubMedBERT) | pubmedbert_cache.pt | 768-dimensional, generated from entity names | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| Node features (Node2Vec) | features_cache.pt (per-seed) | 128-dimensional, trained on train edges only | `data/node_features.py` |
| scRNA-seq (mouse IVC) | AnnData (anndata_processed.h5ad) | 21,230 cells, 8 cell types, IVC stenosis Day 14 + Sham | In-house 10x Genomics; GEO accession pending |
| Human transcriptomics | GSE48000 | 132 samples (107 VTE, 25 controls), whole blood | GEO public dataset |
| scRNA expression export | gnn_network_expression.csv | 272 rows (17 genes × 8 cell types × 2 conditions) | `scripts/export_scRNA_for_R.py` |
| UMAP coordinates | umap_coords.csv | 21,230 cells | `scripts/export_scRNA_for_R.py` |
| Human DE results | GSE48000_de_results.csv | 31,410 genes ranked by logFC | `scripts/render_figure5_cross_species.R` |
| Vein wall fibroblast program | vein_wall_fibroblast_program.csv | 93 human genes | Derived from `DEG_F2rl1pos_Activated_Fib.csv` (top-100, F2RL1 excluded) |

## Table S2: Analysis Script-to-Figure Mapping

| Figure | Rendering Script | Input Data | Release Tag |
|--------|-----------------|------------|-------------|
| Figure 1 | `scripts/render_figure1_kg_temporal.R` | `data/processed/heterodata.pt` | v2.1 |
| Figure 2 | `scripts/render_figure2_benchmark.R` | `checkpoints/full_training_v2/summary.json`, `data/baselines/baseline_results.json` | v2.1 |
| Figure 3 | `scripts/render_figure3_target_ranking.R` | `figures/hidden_targets/full_ranked_candidates.json` | v2.1 |
| Figure 4 | `scripts/render_figure4_scRNA_mapping.R` | `figures/scRNA/gnn_network_expression.csv`, `figures/scRNA/umap_coords.csv` | v2.1 |
| Figure 5 | `scripts/render_figure5_cross_species.R` | `data/GSE48000_de_results.csv`, `data/vein_wall_fibroblast_program.csv` | v2.1 |

## Table S3: Model Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Tempered Heterogeneous Graph Transformer (Tempered HGT) |
| Layers | 2 |
| Attention heads | 4 |
| Hidden dimension | 128 |
| Input features | PubMedBERT 768d + Node2Vec 128d = 896d |
| Learning rate | 5 × 10⁻³ |
| Epochs | 100 (early stopping patience = 15) |
| Batch size | 256 |
| Neighbor sampling | [10, 5, 5] |
| Random seeds | 42, 123, 456, 789, 1024 |
| Baseline models | TransE, DistMult, ComplEx, RotatE |
| Training framework | PyTorch 2.3 + PyTorch Geometric 2.5 |
| Git commit (release) | 8b5caa4 |
| Data hash (heterodata.pt) | Included in `checkpoints/full_training_v2/summary.json` |

## Table S4: Per-Seed Test Performance of Tempered HGT

| Seed | Best Epoch | Val MRR | Test AUROC | Test MRR | Test Hits@10 |
|------|-----------|---------|-----------|----------|--------------|
| 42 | 48 | 0.178 | 0.765 | 0.105 | 0.242 |
| 123 | 14 | 0.059 | 0.612 | 0.051 | 0.071 |
| 456 | 71 | 0.104 | 0.806 | 0.062 | 0.229 |
| 789 | 35 | 0.072 | 0.749 | 0.089 | 0.183 |
| 1024 | 42 | 0.108 | 0.772 | 0.120 | 0.196 |
| Mean | — | 0.104 | 0.741 | 0.086 | 0.184 |
| SD | — | 0.046 | 0.075 | 0.029 | 0.068 |

Source: `checkpoints/full_training_v2/summary.json` (best-epoch checkpoint per seed, evaluated on the held-out test split).

## Table S5: Baseline Model Comparison (Filtered Test Set)

| Model | Filtered MRR | Tail Hits@10 | Head Hits@10 |
|-------|--------------|--------------|--------------|
| TransE | 0.017 | 0.068 | 0.015 |
| DistMult | 0.032 | 0.054 | 0.060 |
| ComplEx | 0.017 | 0.044 | 0.008 |
| RotatE | 0.035 | 0.060 | 0.060 |
| Tempered HGT | 0.086 | 0.184 | 0.184 |

Source: `data/baselines/baseline_results.json`. Tempered HGT reports a single Hits@10 for head and tail under the filtered symmetric evaluation protocol used for the GNN; knowledge-graph embedding baselines are reported separately for head and tail ranking.

## Table S6: Top-30 GNN-Prioritized Candidate Targets

| Rank | Target | Type | Relation | GNN Score |
|------|--------|------|----------|-----------|
| 1 | smad4 | Protein | CONTRIBUTES_TO | 89.6 |
| 2 | san gene programme | Gene | INHIBITS | 75.5 |
| 3 | vwf transcription | Gene | INHIBITS | 70.2 |
| 4 | ace2 | Protein | CONTRIBUTES_TO | 69.0 |
| 5 | tlr4 | Protein | ASSOCIATED_WITH | 64.8 |
| 6 | p-selectin | Protein | CONTRIBUTES_TO | 63.5 |
| 7 | pomc gene transcription | Gene | PROMOTES | 62.8 |
| 8 | tlr2 | Protein | CONTRIBUTES_TO | 62.0 |
| 9 | factor v leiden mutation | Gene | INHIBITS | 59.6 |
| 10 | enos | Protein | CONTRIBUTES_TO | 59.4 |
| 11 | dnmt3a | Gene | INHIBITS | 56.9 |
| 12 | tissue factor | Protein | ASSOCIATED_WITH | 56.0 |
| 13 | mir-155 | Gene | INHIBITS | 55.1 |
| 14 | vegfr2 | Protein | ASSOCIATED_WITH | 51.5 |
| 15 | prolidase | Protein | ASSOCIATED_WITH | 50.5 |
| 16 | thrombin | Protein | CONTRIBUTES_TO | 50.1 |
| 17 | gata2 | Gene | PROMOTES | 49.9 |
| 18 | catalytic subunit crucial to gpi biosynthesis | Protein | ASSOCIATED_WITH | 49.8 |
| 19 | egfr | Protein | CONTRIBUTES_TO | 47.6 |
| 20 | factor x | Protein | CONTRIBUTES_TO | 47.1 |
| 21 | nf-kb | Protein | CONTRIBUTES_TO | 46.4 |
| 22 | tissue factor gene expression under hypoxia | Gene | INHIBITS | 45.0 |
| 23 | runx2 | Gene | PROMOTES | 44.5 |
| 24 | prkca | Gene | CONTRIBUTES_TO | 44.4 |
| 25 | factor xa (fxa) or thrombin | Protein | CONTRIBUTES_TO | 44.0 |
| 26 | fxa | Protein | ASSOCIATED_WITH | 43.2 |
| 27 | largest upregulated gene co-expression module | Gene | INHIBITS | 42.6 |
| 28 | ace2 expression | Gene | PROMOTES | 42.5 |
| 29 | vegfr | Protein | CONTRIBUTES_TO | 42.4 |
| 30 | pai-1 | Protein | CONTRIBUTES_TO | 42.1 |

Source: `figures/hidden_targets/full_ranked_candidates.json` (global scoring of 706,523,994 candidate pairs, no anchor filtering). Target names are as resolved by the knowledge-graph entity vocabulary; `nf-kb` normalization applied to match Figure 3.

## Table S7: Single-Cell RNA-seq Dataset Composition

| Cell Type | Control (n) | DVT (n) | Total (n) | % of Total |
|-----------|-------------|---------|-----------|------------|
| Fibroblast | 2,477 | 9,664 | 12,141 | 57.2% |
| Endothelial | 3,210 | 1,639 | 4,849 | 22.8% |
| VSMC | 22 | 1,099 | 1,121 | 5.3% |
| Monocyte | 3 | 912 | 915 | 4.3% |
| B_cell | 104 | 618 | 722 | 3.4% |
| Macrophage | 1 | 696 | 697 | 3.3% |
| Neutrophil | 49 | 609 | 658 | 3.1% |
| Erythrocyte | 107 | 20 | 127 | 0.6% |
| **Total** | **5,973** | **15,257** | **21,230** | **100.0%** |

Source: `figures/scRNA/umap_coords.csv`. Mouse inferior vena cava (IVC) stenosis model, Day 14 post-surgery versus Sham. Control-group cell numbers are extremely sparse for VSMC, Monocyte, and Macrophage (n = 22, 3, and 1), so differential-expression statistics for these populations were not reported (see Methods).

## Table S8: Key Result Metrics (Mean ± SD across 5 seeds)

| Metric | Value |
|--------|-------|
| Test AUROC | 0.741 ± 0.075 |
| Filtered MRR | 0.086 ± 0.029 |
| Filtered Hits@10 | 0.184 ± 0.068 |
| RotatE Filtered MRR (best baseline) | 0.035 |
| TemperedHGT vs RotatE improvement | 2.5× |
