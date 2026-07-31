"""Export scRNA expression data from AnnData to CSV for R figure generation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SCRNA_DIR = BASE.parent / "figures" / "scRNA"  # scRNA data is in 新思路/figures/scRNA/

ad = sc.read(SCRNA_DIR / "anndata_processed.h5ad")
print(f"Cells: {ad.n_obs:,}, Genes: {ad.n_vars:,}")
print(f"Cell types: {sorted(ad.obs.cell_type.unique())}")

# GNN network genes (mouse orthologs)
gnn_genes = {
    "Inflammation_Program": ["Tlr4", "Nfkb1", "Rela", "Stat3", "Tgfb1", "Spp1", "Tnf", "Il6", "Ccl2", "Sell"],
    "Fibrosis_Program": ["Smad4", "Runx2", "Dnmt3a", "Prkca", "Fn1", "Col1a1", "Acta2"],
}

all_genes = list(set(sum(gnn_genes.values(), [])))
all_genes = [g for g in all_genes if g in ad.var_names]
print(f"GNN network genes found: {len(all_genes)}/{len(set(sum(gnn_genes.values(), [])))}")

rows = []
for ct in sorted(ad.obs.cell_type.unique()):
    for cond in ["Control", "DVT"]:
        sub = ad[(ad.obs.cell_type == ct) & (ad.obs.condition == cond)]
        n = sub.n_obs
        for gene in all_genes:
            expr = sub[:, gene].X
            if hasattr(expr, "toarray"):
                expr = expr.toarray()
            expr_flat = expr.flatten()
            mean_expr = float(np.mean(expr_flat))
            # Mean expression in positive (non-zero) cells only
            pos_expr = expr_flat[expr_flat > 0]
            mean_expr_pos = float(np.mean(pos_expr)) if len(pos_expr) > 0 else 0.0
            pct = float((expr_flat > 0).mean() * 100)
            rows.append({
                "cell_type": ct, "condition": cond, "n_cells": n,
                "gene": gene, "mean_expr": round(mean_expr, 6),
                "mean_expr_pos": round(mean_expr_pos, 6),
                "pct_positive": round(pct, 2),
            })

df = pd.DataFrame(rows)
df.to_csv(SCRNA_DIR / "gnn_network_expression.csv", index=False)
print(f"Exported expression: {len(rows)} rows, {len(all_genes)} genes")

# UMAP coordinates
umap_df = pd.DataFrame({
    "UMAP1": ad.obsm["X_umap"][:, 0],
    "UMAP2": ad.obsm["X_umap"][:, 1],
    "cell_type": ad.obs.cell_type.values,
    "condition": ad.obs.condition.values,
})
umap_df.to_csv(SCRNA_DIR / "umap_coords.csv", index=False)
print(f"Exported UMAP: {len(umap_df)} cells")

# Cell proportion data (for reference)
prop_rows = []
for ct in sorted(ad.obs.cell_type.unique()):
    for cond in ["Control", "DVT"]:
        n = int((ad.obs.cell_type == ct).sum())
        n_cond = int(((ad.obs.cell_type == ct) & (ad.obs.condition == cond)).sum())
        prop_rows.append({"cell_type": ct, "condition": cond, "n_cells": n_cond})

prop_df = pd.DataFrame(prop_rows)
prop_df.to_csv(SCRNA_DIR / "cell_proportion.csv", index=False)
print(f"Exported cell proportions")

print("Done. Ready for R figure generation.")
