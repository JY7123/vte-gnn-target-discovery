# Code Availability Statement

*For inclusion in the Nature Communications manuscript under the "Code Availability" section.*

---

All custom code used in this study is publicly available at Zenodo with the DOI 10.5281/zenodo.21757258 and on GitHub at https://github.com/JY7123/vte-gnn-target-discovery.

The repository includes:

1. **Complete Python implementation** of the Tempered Heterogeneous Graph Transformer (`models/tempered_hgt.py`), including the learnable per-relation temperature mechanism (τ), three-layer prior injection system, and attention-guided explainability pipeline.

2. **Reproduction scripts** (`reproduce.sh` / `reproduce.bat`) that execute the full pipeline from pre-processed knowledge graph data to paper figures in a single command.

3. **Pre-processed data** (`heterodata.pt`, ~11 MB; PCA-compressed PubMedBERT features `features_128d.pt`, ~40 MB) sufficient to reproduce all model training, evaluation, and figure generation results without requiring access to the source Neo4j database.

4. **124 unit and integration tests** (pytest) covering data pipeline integrity, model correctness, temporal split validation, literature novelty classification, and ablation consistency.

5. **R code** for Figure 2 generation (`render_figure2.R`) using ggplot2 and patchwork.

**Software dependencies** are fully specified in `environment.yml` (conda). The core computational environment consists of Python 3.12, PyTorch 2.3, PyTorch Geometric 2.5, and the HuggingFace Transformers library. The Tempered HGT model was trained on an NVIDIA RTX 5060 (8 GB VRAM) but is compatible with CPU-only execution (~40 minutes per model on a modern CPU).

**Third-party software used:**
- Neo4j v5.26 (knowledge graph storage; not required for reproduction from shared HeteroData)
- PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
- NCBI E-utilities API (PMID date resolution)
- R v4.5.2 with packages: ggplot2, dplyr, tidyr, jsonlite, patchwork, scales
- CellChat v2 (scRNA-seq ligand-receptor analysis)
- dorothea/viper (transcription factor activity inference)
- Seurat v5 (scRNA-seq preprocessing)