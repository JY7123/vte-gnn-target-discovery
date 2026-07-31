# Supplementary Materials

---

## Table S1: Data Lineage and Version Tracking

| Resource | Version / Identifier | Description | Source |
|----------|---------------------|-------------|--------|
| Knowledge Graph | v2.0 (Zenodo DOI: 10.5281/zenodo.21724152) | 82,644 nodes, 14 entity types, 29 curated edge types, 11,989 edges | Built from PubMed abstracts + PMC full-text + curated databases |
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

### Analysis Script-to-Figure Mapping

| Figure | Rendering Script | Input Data | Commit Hash |
|--------|-----------------|------------|-------------|
| Figure 1 | `scripts/render_figure1_kg_temporal.R` | `data/processed/heterodata.pt` | 52a3f3d |
| Figure 2 | `scripts/render_figure2_benchmark.R` | `checkpoints/full_training_v2/summary.json`, `data/baselines/baseline_results.json` | 52a3f3d |
| Figure 3 | `scripts/render_figure3_target_ranking.R` | `figures/hidden_targets/full_ranked_candidates.json` | 52a3f3d |
| Figure 4 | `scripts/render_figure4_scRNA_mapping.R` | `figures/scRNA/gnn_network_expression.csv`, `figures/scRNA/umap_coords.csv` | 52a3f3d |
| Figure 5 | `scripts/render_figure5_cross_species.R` | `data/GSE48000_de_results.csv` | 52a3f3d |
| Supplementary Figure S1 | `render_supp_fig1.R` | `checkpoints/full_training_v2/summary.json` | 52a3f3d |

### Model Training Configuration

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
| Training framework | PyTorch 2.12 + PyTorch Geometric |
| Git commit (release) | 52a3f3d |
| Data hash (heterodata.pt) | Included in `checkpoints/full_training_v2/summary.json` |

### Key Result Metrics (Mean ± SD across 5 seeds)

| Metric | Value |
|--------|-------|
| Test AUROC | 0.741 ± 0.075 |
| Filtered MRR | 0.086 ± 0.029 |
| Filtered Hits@1 | 0.037 ± 0.019 |
| Filtered Hits@3 | 0.090 ± 0.036 |
| Filtered Hits@10 | 0.184 ± 0.068 |
| RotatE Filtered MRR (best baseline) | 0.035 |
| TemperedHGT vs RotatE improvement | 2.5× |
