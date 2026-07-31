# VTE GNN Target Discovery — Code Availability

This directory contains the complete codebase for reproducing results reported in:

> **"Temporally Evaluated Heterogeneous Graph Learning Prioritizes Cell-Type-Specific Inflammatory and Fibrotic Programs in Venous Thromboembolism"**

---

## System Requirements

- **OS**: Linux, Windows, or macOS
- **RAM**: ≥ 32 GB recommended (16 GB minimum)
- **GPU**: Optional (CPU training supported, ~40 min per model on modern CPU)
- **Python**: 3.12
- **R**: 4.4+ (for Figure 2 only; packages: ggplot2, dplyr, tidyr, jsonlite, patchwork, scales)

## Quick Start (5 minutes)

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate vte-gnn

# 2. Download pre-processed data (Zenodo, ~52 MB)
#    Place in data/processed/:
#    - heterodata.pt (knowledge graph, 82,644 nodes × 248,240 edges)
#    - features_128d.pt (PCA-compressed PubMedBERT embeddings)
#    OR use the reproduction script below to regenerate from source.

# 3. Run tests to verify installation
pytest tests/ -x -q

# 4. One-click reproduction
bash reproduce.sh        # Linux/macOS
reproduce.bat            # Windows
```

## Reproducing Results

### From shared data (recommended)

The pre-processed knowledge graph (`heterodata.pt`) and PCA features (`features_128d.pt`) are available on Zenodo at 10.5281/zenodo.21225701. Download and place them in `data/processed/`, then:

```bash
# Train the main Tempered HGT model (PCA 128d features)
python train_full_v2.py

# Run KGE baselines (TransE, DistMult, ComplEx, RotatE)
python scripts/run_baselines_full.py

# Generate hidden target predictions
python hidden_target_hunter.py

# Render all paper figures (Figures 1-5)
Rscript scripts/render_figure1_kg_temporal.R
Rscript scripts/render_figure2_benchmark.R
Rscript scripts/render_figure3_target_ranking.R
Rscript scripts/render_figure4_scRNA_mapping.R
Rscript scripts/render_figure5_cross_species.R
```

### From raw Neo4j knowledge graph (full reproduction)

If you have access to the Neo4j database:

```bash
# 1. Export KG from Neo4j → PyG HeteroData
python data/neo4j_to_pyg.py

# 2. Enrich PMID publication dates
python data/pmid_date_lookup.py

# 3. Generate features (PubMedBERT 768d → PCA 128d)
python data/node_features.py

# 4. Negative sampling + temporal split
python data/negative_sampling.py
python data/temporal_split.py

# 5. Build dataset
python data/build_dataset.py

# 6. Train + evaluate + render (same as above)
python train_full_v2.py
python render_paper_figures.py
```

## Repository Structure

```
vte_gnn_target_discovery/
├── config/                     # YAML configuration files
│   ├── anchor_config.yaml      # Biological prior: anchor genes, cascade, edge multipliers
│   └── ablation_config.yaml    # Ablation mode definitions
│
├── data/                       # Data pipeline modules
│   ├── neo4j_to_pyg.py         # Neo4j → PyG HeteroData exporter (multi-label aware)
│   ├── node_features.py        # PubMedBERT + Node2Vec feature generation
│   ├── negative_sampling.py    # Degree-preserving + hard negative sampling
│   ├── temporal_split.py       # PMID-based prospective temporal splitting
│   ├── pmid_date_lookup.py     # NCBI E-utilities PMID → date resolution
│   ├── build_dataset.py        # Assemble train/val/test splits
│   ├── ablation_injection.py   # False positive injection for ablation
│   └── processed/              # ← Place downloaded .pt files here
│
├── models/                     # Neural network architectures
│   ├── tempered_hgt.py         # TemperedHGT: core model with learnable τ
│   ├── encoders.py             # PubMedBERT encoder + InnerProductDecoder
│   └── baselines.py            # RGCN, HAN baseline implementations
│
├── training/                   # Training infrastructure
│   ├── link_prediction.py      # LinkPredictionTrainer (BCE + early stopping)
│   ├── baseline_trainer.py     # FairTrainer: unified hyperparameter-locked training
│   ├── edge_bias.py            # CosineAnnealingDecay + EdgeBiasInitializer
│   └── metrics.py              # AUROC, MRR, Hits@K computation
│
├── explainability/             # Model interpretation
│   ├── gnnexplainer_vte.py     # VTEExplainer: attention-based GNN explanation
│   ├── alignment_engine.py     # AnchorAlignmentEngine: cascade step mapping
│   ├── contradiction_gate.py   # Path contradiction detection
│   └── subgraph_extractor.py   # JSON/CSV → Cytoscape export
│
├── validation/                 # External validation
│   ├── literature_validation.py # PubMed novelty classification
│   ├── cross_check_mr.py       # Mendelian randomization cross-validation
│   ├── error_correction.py     # Prior error correction quantification
│   └── aggregate_results.py    # Comparison tables + LaTeX export
│
├── tests/                      # 124 unit + integration tests (pytest)
│   ├── test_tempered_hgt.py    # Core model correctness
│   ├── test_neo4j_to_pyg.py    # Exporter multi-label handling
│   ├── test_temporal_split.py  # Prospective validation integrity
│   ├── test_literature_validation.py
│   └── ... (20 test files)
│
├── train_full_v2.py            # Main training entry point
├── hidden_target_hunter.py     # Target discovery + ranking pipeline
├── render_paper_figures.py     # Figure 1, 3, 4, 5 generation (Python)
├── render_figure2.R            # Figure 2 generation (R/ggplot2)
├── reproduce.sh / reproduce.bat # One-click reproduction
├── environment.yml             # Conda environment specification
└── README.md                   # This file
```

## Key Model: Tempered HGT

The core contribution is the **Tempered Heterogeneous Graph Transformer**, defined in `models/tempered_hgt.py`.

**Core attention formula:**
```
α = softmax(QKᵀ / (τ · √d) + b_edge · cos_decay)
```

- **τ (tau)**: Per-relation learnable temperature, stored in `nn.ParameterDict`. Each relation type `(src, rel, dst)` receives its own τ, initialized at 1.0 and optimized during training.
- **b_edge**: Edge weight bias encoding curated biological prior from `config/anchor_config.yaml`.
- **cos_decay**: Cosine annealing schedule decaying from 1.0 → 0.0 over training. The hard prior exerts strongest influence early, then yields to data-driven learning.

**Three-layer prior injection:**
1. **Layer 1 (Hard)**: Edge weight biases provide structured biological initialization
2. **Layer 2 (Soft)**: Per-relation τ autonomously suppresses noisy relation types (emergent)
3. **Layer 3 (Interpretability)**: Attention-guided BFS for mechanism cascade mapping

## Key Results (Leakage-Free Evaluation, Mean ± SD across 5 seeds)

| Model | Test AUROC | Filtered MRR | Filtered Hits@10 |
|-------|-----------|-------------|-----------------|
| **Tempered HGT** | **0.741 ± 0.075** | **0.086 ± 0.029** | **0.184 ± 0.068** |
| RotatE (best KGE baseline) | — | 0.035 | 0.060 |
| TransE | — | 0.017 | 0.068 |
| DistMult | — | — | 0.054 |
| ComplEx | — | 0.017 | — |
| **TemperedHGT vs RotatE** | — | **2.5× improvement** | **3.1× improvement** |

Top-ranked target: **SMAD4** (Rank #1, entity-resolved).

## Tests

```bash
pytest tests/ -v           # Full test suite (124 pass, 2 skip)
pytest tests/ -x -q        # Quick smoke test
```

## License

This code is provided for academic and research purposes. See LICENSE file for details.

## Citation

If you use this code, please cite:

> [Authors]. Temporally Evaluated Heterogeneous Graph Learning Prioritizes Cell-Type-Specific Inflammatory and Fibrotic Programs in Venous Thromboembolism. (2026).

The code is archived at Zenodo: 10.5281/zenodo.21225701 (v1.0) and will be updated for v2.0.
