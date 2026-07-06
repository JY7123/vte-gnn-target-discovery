#!/bin/bash
# reproduce.sh — One-click reproduction pipeline for VTE GNN Target Discovery
# Requires: conda environment 'vte-gnn' (see environment.yml)
#           Pre-processed data in data/processed/ (heterodata.pt + features_128d.pt)
set -e

echo "============================================"
echo "VTE GNN Target Discovery — Reproduction"
echo "============================================"

# ── 1. Verify environment ──
echo ""
echo "[1/5] Verifying environment..."
python -c "import torch; print(f'  PyTorch {torch.__version__}')"
python -c "import torch_geometric; print(f'  PyG {torch_geometric.__version__}')"

# ── 2. Verify data ──
echo ""
echo "[2/5] Verifying data files..."
for f in data/processed/heterodata.pt data/processed/train_edges.pt data/processed/negative_edges.pt; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f — please download from Zenodo [DOI]"
        exit 1
    fi
done
if [ -f "checkpoints/pca_features/features_128d.pt" ]; then
    echo "  [OK] PCA features found"
else
    echo "  [INFO] PCA features not found — will use random 128d features"
fi

# ── 3. Run tests ──
echo ""
echo "[3/5] Running tests..."
pytest tests/ -x -q --tb=short 2>&1 | tail -5

# ── 4. Train model ──
echo ""
echo "[4/5] Training Tempered HGT (PCA 128d, 93 epochs)..."
python train_full_v2.py

# ── 5. Generate figures ──
echo ""
echo "[5/5] Generating paper figures..."
python render_paper_figures.py
echo "  Python figures complete."

# Figure 2 in R (if Rscript available)
if command -v Rscript &> /dev/null; then
    Rscript render_figure2.R
    echo "  Figure 2 (R) complete."
else
    echo "  [SKIP] Rscript not found — Figure 2 can be generated separately with: Rscript render_figure2.R"
fi

echo ""
echo "============================================"
echo "Reproduction complete."
echo "Figures saved to figures/paper_figures/"
echo "============================================"
