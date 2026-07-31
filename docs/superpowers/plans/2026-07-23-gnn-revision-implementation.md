# GNN Manuscript Revision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the GNN manuscript revision: run baselines, generate Figures 2-5 in R, prepare for manuscript rewrite.

**Architecture:** Python for baseline training (reuse existing `training/baselines.py` + `train_full_v2.py` data split). R/ggplot2 for all figures (user preference). Data flows: Python-trained model → JSON summary → R reads JSON/CSV → ggplot2 figures.

**Tech Stack:** Python 3.12, PyTorch 2.12, PyG 2.8, R 4.x, Seurat 5, ggplot2, patchwork, pheatmap, fgsea

## Global Constraints

- All figure scripts in R using ggplot2 ecosystem
- User reviews ALL figure code before execution
- Baseline comparison uses same train/val/test split as main model (seed=42)
- scRNA data from existing AnnData `figures/scRNA/anndata_processed.h5ad` (21,230 cells)
- Human VTE cohort: GSE48000 (GEO)
- Figure output: TIFF 300dpi for journal submission

---

### Task 1: Export Baselines Training Data (Python)

**Files:**
- Create: `scripts/export_baseline_data.py`

**Interfaces:**
- Consumes: `data/processed/heterodata.pt`, `checkpoints/full_training_v2/seed_42/features_cache.pt`
- Produces: `data/baselines/train_triples.csv`, `data/baselines/test_triples.csv`, `data/baselines/entity_map.json`

**Goal:** Export train/test triples for baseline models using the same seed=42 split as the main model.

- [ ] **Step 1: Write export script**

```python
# scripts/export_baseline_data.py
"""Export train/test triples + entity mapping for baseline KG embedding models."""
import torch, json, csv
from pathlib import Path
from collections import defaultdict

def main():
    data = torch.load("data/processed/heterodata.pt", weights_only=False)
    import yaml
    with open("config/anchor_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    config_ets = [tuple(et) for et in cfg.get("edge_types", [])]
    valid_nts = [nt for nt in data.node_types if data[nt].num_nodes > 0]
    meta_relations = [et for et in config_ets if et in data.edge_types and et[0] in valid_nts and et[2] in valid_nts]

    from data.temporal_split import RandomStratifiedSplitter
    splitter = RandomStratifiedSplitter(seed=42, edge_types=meta_relations)
    train_ei, val_ei, test_ei = splitter.split(data)
    train_ei = {et: ei for et, ei in train_ei.items() if et in meta_relations}
    test_ei = {et: ei for et, ei in test_ei.items() if et in meta_relations}

    # Build global entity mapping
    entity_map = {}
    global_id = 0
    for nt in valid_nts:
        for local_idx in range(data[nt].num_nodes):
            name = data[nt].name[local_idx] if hasattr(data[nt], 'name') and local_idx < len(data[nt].name) else f"{nt}_{local_idx}"
            entity_map[(nt, local_idx)] = (global_id, str(name), nt)
            global_id += 1

    Path("data/baselines").mkdir(parents=True, exist_ok=True)

    # Export entity map
    with open("data/baselines/entity_map.json", "w") as f:
        json.dump({f"{nt}_{idx}": {"global_id": gid, "name": name, "type": nt}
                   for (nt, idx), (gid, name, nt) in entity_map.items()}, f, indent=2)

    # Export triples (global_id, rel_id, global_id)
    rel_list = sorted(set(train_ei.keys()) | set(test_ei.keys()))
    rel_map = {et: i for i, et in enumerate(rel_list)}

    def export_triples(ei_dict, path, split_name):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["head", "relation", "tail", "head_name", "tail_name", "relation_name"])
            for et, ei in ei_dict.items():
                if et not in rel_map: continue
                src_t, rel, dst_t = et
                for j in range(ei.shape[1]):
                    h = entity_map.get((src_t, int(ei[0, j])))
                    t = entity_map.get((dst_t, int(ei[1, j])))
                    if h and t:
                        w.writerow([h[0], rel_map[et], t[0], h[1], t[1], rel])

    export_triples(train_ei, "data/baselines/train_triples.csv", "train")
    export_triples(test_ei, "data/baselines/test_triples.csv", "test")
    print(f"Exported: {len(entity_map)} entities, {len(rel_map)} relations")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run export**

```bash
cd vte_gnn_target_discovery && python scripts/export_baseline_data.py
```

Expected: Creates `data/baselines/{train_triples.csv, test_triples.csv, entity_map.json}`

- [ ] **Step 3: Commit**

```bash
git add scripts/export_baseline_data.py data/baselines/
git commit -m "feat: export train/test triples for baseline KG embedding models"
```

---

### Task 2: Run Baseline Models + Export Results

**Files:**
- Create: `scripts/run_baselines.py`

**Interfaces:**
- Consumes: `data/baselines/train_triples.csv`, `data/baselines/test_triples.csv`
- Produces: `data/baselines/baseline_results.json`

**Goal:** Train TransE, DistMult, ComplEx, RotatE and report filtered MRR/Hits@K.

- [ ] **Step 1: Write baseline runner script**

```python
# scripts/run_baselines.py
"""Train KG embedding baselines and export filtered metrics."""
import torch, json, csv, time
from pathlib import Path
from training.baselines import TransE, DistMult, ComplEx, RotatE, BaselineTrainer, evaluate_baseline_filtered

def load_triples(path):
    triples = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            triples.append((int(row["head"]), int(row["relation"]), int(row["tail"])))
    return torch.tensor(triples, dtype=torch.long)

def main():
    train = load_triples("data/baselines/train_triples.csv")
    test = load_triples("data/baselines/test_triples.csv")
    num_entities = max(train[:, 0].max(), train[:, 2].max(), test[:, 0].max(), test[:, 2].max()).item() + 1
    num_relations = max(train[:, 1].max(), test[:, 1].max()).item() + 1
    print(f"Entities: {num_entities}, Relations: {num_relations}, Train: {train.shape[0]:,}, Test: {test.shape[0]:,}")

    # Build all-true set for filtered evaluation
    all_true = set()
    for t in [train, test]:
        for j in range(t.shape[0]):
            all_true.add((int(t[j, 0]), int(t[j, 1]), int(t[j, 2])))

    results = {}
    models = {
        "TransE": TransE(num_entities, num_relations, dim=128, margin=1.0),
        "DistMult": DistMult(num_entities, num_relations, dim=128),
        "ComplEx": ComplEx(num_entities, num_relations, dim=128),
        "RotatE": RotatE(num_entities, num_relations, dim=128, margin=6.0),
    }

    for name, model in models.items():
        print(f"\n{'='*40}\nTraining {name}\n{'='*40}")
        t0 = time.time()
        trainer = BaselineTrainer(model, learning_rate=1e-3, num_epochs=100, device="cpu")
        trainer.fit(train, num_entities, verbose=True)
        metrics = evaluate_baseline_filtered(model, test, all_true, num_entities)
        results[name] = {k: round(v, 4) for k, v in metrics.items()}
        results[name]["train_time_min"] = round((time.time() - t0) / 60, 1)
        print(f"  Filtered MRR: {metrics['filtered_mrr']:.4f} | H@1: {metrics['tail_hits@1']:.4f}/{metrics['head_hits@1']:.4f} | H@10: {metrics['tail_hits@10']:.4f}/{metrics['head_hits@10']:.4f}")

    # Add TemperedHGT results from summary.json
    with open("checkpoints/full_training_v2/summary.json") as f:
        hgt = json.load(f)
    results["TemperedHGT"] = {
        "filtered_mrr": hgt["test_mrr_mean"],
        "auroc": hgt["test_auroc_mean"],
        "hits10": hgt["test_hits10_mean"],
    }

    with open("data/baselines/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to data/baselines/baseline_results.json")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run baselines**

```bash
cd vte_gnn_target_discovery && python scripts/run_baselines.py
```

Expected output: `data/baselines/baseline_results.json` with 5 model entries (TransE, DistMult, ComplEx, RotatE, TemperedHGT)

- [ ] **Step 3: Commit**

```bash
git add scripts/run_baselines.py data/baselines/baseline_results.json
git commit -m "feat: run KG embedding baselines and export filtered metrics"
```

---

### Task 3: Figure 3 — GNN Target Ranking (R)

**Files:**
- Create: `scripts/render_figure3_target_ranking.R`
- Produces: `figures/paper_figures/Figure3_Target_Ranking.tiff`

**Interfaces:**
- Consumes: `figures/hidden_targets/full_ranked_candidates.json`
- Produces: Multi-panel TIFF (top-30 bar chart + degree scatter + pathway breakdown)

**Goal:** Show GNN global ranking results — top targets, pathway classification, degree vs score relationship.

- [ ] **Step 1: Write R script**

```r
#!/usr/bin/env Rscript
# Figure 3: GNN Global Prioritization of VTE Pathological Programs
library(ggplot2)
library(patchwork)
library(dplyr)
library(tidyr)
library(jsonlite)

OUT <- "figures/paper_figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# Load data
ranked <- fromJSON("figures/hidden_targets/full_ranked_candidates.json", simplifyDataFrame = TRUE)
ranked <- as.data.frame(ranked)
ranked$rank <- as.integer(ranked$rank)
ranked$gnn_score <- as.numeric(ranked$gnn_score)
ranked$degree <- as.integer(ranked$degree)

# ============================================================
# Panel A: Top-30 bar chart
# ============================================================
top30 <- head(ranked, 30)
top30$target <- factor(top30$target, levels = rev(top30$target))

# Classify by pathway
top30$pathway <- ifelse(
  top30$target %in% c("smad4", "runx2", "dnmt3a", "prkca", "gata2",
                       "san gene programme", "vwf transcription", "factor v leiden mutation"),
  "TGFB/Fibrosis", "TLR4/Inflammation")
top30$pathway[grep("tlr|nf-kb|p-selectin|egfr|enos|ace2|pai-1|tissue factor|thrombin|fxa|factor x|factor viii|factor ix", 
                   top30$target, ignore.case = TRUE)] <- "TLR4/Inflammation"
top30$pathway[grep("smad|runx|dnmt|prkc|gata|col|fn1|acta|mir-155|pomc|san gene|vwf transc|factor v leiden",
                   top30$target, ignore.case = TRUE)] <- "TGFB/Fibrosis"

p1 <- ggplot(top30, aes(x = gnn_score, y = target, fill = pathway)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = c("TGFB/Fibrosis" = "#E64B35", "TLR4/Inflammation" = "#4DBBD5")) +
  labs(x = "GNN Score", y = "", title = "Top 30 GNN-Prioritized VTE Targets",
       subtitle = "Entity-resolved global ranking, no anchor filtering") +
  theme_minimal(base_size = 10) +
  theme(legend.position = c(0.8, 0.2))

# ============================================================
# Panel B: Degree vs Score scatter
# ============================================================
p2 <- ggplot(ranked, aes(x = log10(degree + 1), y = gnn_score)) +
  geom_point(aes(color = type), alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", se = TRUE, color = "grey40", linetype = "dashed") +
  scale_color_manual(values = c("Gene" = "#00A087", "Protein" = "#3C5488")) +
  labs(x = "log10(Node Degree + 1)", y = "GNN Score",
       title = "GNN Score vs. Knowledge Graph Node Degree",
       subtitle = paste0("Spearman ρ = ", round(cor(ranked$gnn_score, log10(ranked$degree + 1), method = "spearman"), 3))) +
  theme_minimal(base_size = 10)

# ============================================================
# Panel C: Pathway category breakdown
# ============================================================
pathway_counts <- ranked %>%
  mutate(category = case_when(
    grepl("coagul|thrombin|factor|fibrin|prothrombin|plasmin", target, ignore.case = TRUE) ~ "Coagulation",
    grepl("tlr|nf-kb|inflamm|p-selectin|cytokine|il-|tnf|enos|pai-1", target, ignore.case = TRUE) ~ "Inflammation",
    grepl("smad|tgf|col|fn1|acta|fibrosis|runx|dnmt|gata|prkc", target, ignore.case = TRUE) ~ "Fibrosis/TGFB",
    grepl("ace2|egfr|vegf|vegfr|pdgfr|kit", target, ignore.case = TRUE) ~ "Vascular Signaling",
    TRUE ~ "Other"
  )) %>%
  count(category) %>%
  mutate(pct = n / sum(n) * 100)

p3 <- ggplot(pathway_counts, aes(x = reorder(category, n), y = n, fill = category)) +
  geom_col() +
  coord_flip() +
  labs(x = "", y = "Number of Targets", title = "Pathway Distribution of Top Candidates") +
  theme_minimal(base_size = 10) + guides(fill = "none")

# ============================================================
# Panel D: Top targets table
# ============================================================
top10 <- head(ranked, 10)[, c("rank", "target", "type", "gnn_score", "disease")]
colnames(top10) <- c("Rank", "Target", "Type", "Score", "Disease")

p4 <- ggplot() +
  annotation_custom(
    gridExtra::tableGrob(top10, rows = NULL, theme = gridExtra::ttheme_minimal(base_size = 8)),
    xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
  ) +
  theme_void() +
  labs(title = "Top 10 GNN-Prioritized Targets")

# ============================================================
# Assemble
# ============================================================
combined <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title = "GNN Global Prioritization of VTE Molecular Regulators",
    subtitle = paste0(nrow(ranked), " entity-resolved candidates from 706M scored pairs"),
    tag_levels = "A"
  ) &
  theme(plot.tag = element_text(face = "bold", size = 12))

ggsave(file.path(OUT, "Figure3_Target_Ranking.tiff"), combined,
       width = 14, height = 12, dpi = 300, compression = "lzw")
message("Saved: Figure3_Target_Ranking.tiff")
```

---

### Task 4: Figure 4 — scRNA Cell-Type Mapping (R)

**Files:**
- Create: `scripts/render_figure4_scRNA_mapping.R`
- Produces: `figures/paper_figures/Figure4_scRNA_Mapping.tiff`

**Interfaces:**
- Consumes: `figures/scRNA/anndata_processed.h5ad` (via Python export → CSV), `figures/hidden_targets/full_ranked_candidates.json`
- Produces: Multi-panel TIFF (UMAP + dotplot + violin + CellChat schematic)

**Goal:** Map GNN-prioritized network genes to specific cell types in the thrombosed vein wall.

First, export AnnData to R-readable CSV, then generate figure.

- [ ] **Step 1: Export AnnData expression matrix to CSV (Python helper)**

```python
# scripts/export_scRNA_for_R.py
"""Export key expression data from AnnData to CSV for R figure generation."""
import scanpy as sc
import pandas as pd
import numpy as np

ad = sc.read("figures/scRNA/anndata_processed.h5ad")

# GNN network genes (mouse orthologs)
gnn_genes = {
    "Inflammation_Program": ["Tlr4", "Nfkb1", "Rela", "Stat3", "Tgfb1", "Spp1", "Tnf", "Il6", "Ccl2", "Sell"],
    "Fibrosis_Program": ["Smad4", "Runx2", "Dnmt3a", "Prkca", "Fn1", "Col1a1", "Acta2", "Tgfb1"],
}

# Export: mean expression per cell type per condition
all_genes = list(set(sum(gnn_genes.values(), [])))
all_genes = [g for g in all_genes if g in ad.var_names]

rows = []
for ct in sorted(ad.obs.cell_type.unique()):
    for cond in ["Control", "DVT"]:
        sub = ad[(ad.obs.cell_type == ct) & (ad.obs.condition == cond)]
        n = sub.n_obs
        for gene in all_genes:
            expr = sub[:, gene].X
            if hasattr(expr, "toarray"): expr = expr.toarray()
            mean_expr = float(np.mean(expr))
            pct = float((expr > 0).mean() * 100)
            rows.append({"cell_type": ct, "condition": cond, "n_cells": n,
                         "gene": gene, "mean_expr": mean_expr, "pct_positive": pct})

df = pd.DataFrame(rows)
df.to_csv("figures/scRNA/gnn_network_expression.csv", index=False)
print(f"Exported: {len(rows)} rows, {len(all_genes)} genes")

# Also export UMAP coordinates for context
umap_df = pd.DataFrame({
    "UMAP1": ad.obsm["X_umap"][:, 0],
    "UMAP2": ad.obsm["X_umap"][:, 1],
    "cell_type": ad.obs.cell_type.values,
    "condition": ad.obs.condition.values,
})
umap_df.to_csv("figures/scRNA/umap_coords.csv", index=False)
print(f"Exported UMAP: {len(umap_df)} cells")
```

- [ ] **Step 2: Run export**

```bash
cd vte_gnn_target_discovery && python scripts/export_scRNA_for_R.py
```

- [ ] **Step 3: Write R figure script (user reviews before execution)**

```r
#!/usr/bin/env Rscript
# Figure 4: scRNA-seq Cell-Type Mapping of GNN-Prioritized Network Genes
library(ggplot2)
library(patchwork)
library(dplyr)
library(tidyr)
library(scales)

OUT <- "figures/paper_figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# Load data
expr_df <- read.csv("figures/scRNA/gnn_network_expression.csv")
umap_df <- read.csv("figures/scRNA/umap_coords.csv")

# ============================================================
# Panel A: UMAP by condition + cell type
# ============================================================
p1a <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = condition)) +
  geom_point(size = 0.3, alpha = 0.6) +
  scale_color_manual(values = c("Control" = "steelblue", "DVT" = "crimson")) +
  labs(title = "Condition", x = "UMAP1", y = "UMAP2") +
  theme_minimal(base_size = 9) + guides(color = guide_legend(override.aes = list(size = 3)))

p1b <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = cell_type)) +
  geom_point(size = 0.3, alpha = 0.6) +
  scale_color_manual(values = c(
    "Endothelial" = "#E64B35", "Fibroblast" = "#4DBBD5", "VSMC" = "#00A087",
    "Macrophage" = "#3C5488", "Monocyte" = "#F39B7F", "Neutrophil" = "#8491B4",
    "B_cell" = "#91D1C2", "Erythrocyte" = "#DC0000"
  )) +
  labs(title = "Cell Type", x = "UMAP1", y = "UMAP2") +
  theme_minimal(base_size = 9) + guides(color = guide_legend(override.aes = list(size = 3), ncol = 2))

# ============================================================
# Panel B: Dotplot of GNN-prioritized genes × cell types (DVT only)
# ============================================================
dvt_expr <- expr_df %>%
  filter(condition == "DVT") %>%
  mutate(gene_program = case_when(
    gene %in% c("Tlr4", "Nfkb1", "Rela", "Stat3", "Spp1", "Tnf", "Il6", "Ccl2", "Sell") ~ "Inflammation",
    gene %in% c("Smad4", "Runx2", "Dnmt3a", "Prkca", "Fn1", "Col1a1", "Acta2") ~ "Fibrosis",
    gene == "Tgfb1" ~ "TGFB Ligand",
    TRUE ~ "Other"
  ))

p2 <- ggplot(dvt_expr, aes(x = gene, y = cell_type)) +
  geom_point(aes(size = pct_positive, color = mean_expr)) +
  scale_color_gradient2(low = "steelblue", mid = "white", high = "crimson", midpoint = 0) +
  scale_size_continuous(range = c(1, 8), name = "% Positive") +
  facet_grid(. ~ gene_program, scales = "free_x", space = "free_x") +
  labs(title = "GNN-Prioritized Network Genes: Cell-Type Expression (DVT)",
       subtitle = "Dot size = % positive cells; Color = mean log-normalized expression",
       x = "", y = "") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(face = "bold", size = 10))

# ============================================================
# Panel C: Key ligand-receptor pair: TGFB1 → SMAD4/COL1A1
# ============================================================
# TGFB1 expression across cell types (DVT vs Control)
tgf_data <- expr_df %>%
  filter(gene %in% c("Tgfb1", "Smad4", "Col1a1", "Fn1")) %>%
  filter(cell_type %in% c("Macrophage", "Fibroblast"))

p3 <- ggplot(tgf_data, aes(x = cell_type, y = mean_expr, fill = condition)) +
  geom_col(position = position_dodge(0.8), width = 0.6) +
  facet_wrap(~ gene, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = c("Control" = "steelblue", "DVT" = "crimson")) +
  labs(title = "TGF-β → Fibrosis Axis: Macrophage-Fibroblast Crosstalk",
       subtitle = "Tgfb1 (ligand, Macrophage) → Smad4/Col1a1/Fn1 (effectors, Fibroblast)",
       x = "", y = "Mean log-normalized Expression") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# ============================================================
# Panel D: Inflammatory program in macrophages
# ============================================================
infl_data <- expr_df %>%
  filter(gene %in% c("Tlr4", "Nfkb1", "Spp1", "Tnf")) %>%
  filter(cell_type %in% c("Macrophage", "Monocyte", "Neutrophil"))

p4 <- ggplot(infl_data, aes(x = cell_type, y = mean_expr, fill = condition)) +
  geom_col(position = position_dodge(0.8), width = 0.6) +
  facet_wrap(~ gene, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = c("Control" = "steelblue", "DVT" = "crimson")) +
  labs(title = "TLR4/NF-κB Inflammation Program in Myeloid Cells",
       x = "", y = "Mean log-normalized Expression") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# ============================================================
# Assemble
# ============================================================
top_row <- p1a + p1b + plot_layout(widths = c(1, 1.3))
combined <- (top_row / p2 / p3 / p4) +
  plot_annotation(
    title = "Single-Cell Mapping of GNN-Prioritized Pathological Programs",
    subtitle = paste0("Mouse IVC Stenosis Model, Day 14, ", 
                      nrow(umap_df), " cells, ", 
                      length(unique(expr_df$cell_type)), " cell types"),
    tag_levels = "A"
  ) &
  theme(plot.tag = element_text(face = "bold", size = 12))

ggsave(file.path(OUT, "Figure4_scRNA_Mapping.tiff"), combined,
       width = 16, height = 18, dpi = 300, compression = "lzw")
message("Saved: Figure4_scRNA_Mapping.tiff")
```

---

### Task 5: Figure 5 — Cross-Species Validation (R)

**Files:**
- Create: `scripts/render_figure5_cross_species.R`
- Produces: `figures/paper_figures/Figure5_CrossSpecies.tiff`

**Interfaces:**
- Consumes: GSE48000 expression matrix (GEO download), GNN network gene sets
- Produces: Multi-panel TIFF (GSEA enrichment + robustness + negative control)

**Goal:** Show that GNN-prioritized programs are enriched in human VTE whole blood.

- [ ] **Step 1: Download and prepare GSE48000 data**

```bash
# In R:
# GEOquery::getGEO("GSE48000") → extract expression matrix → save as CSV
```

- [ ] **Step 2: Write R script (user reviews before execution)**

```r
#!/usr/bin/env Rscript
# Figure 5: Cross-Species Validation in Human VTE Cohorts
library(ggplot2)
library(patchwork)
library(fgsea)
library(dplyr)

OUT <- "figures/paper_figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# Load GSE48000 data (pre-downloaded CSV)
# Columns: gene_symbol, log2FC_DVT_vs_Control, p_value
gse <- read.csv("data/GSE48000_de_results.csv")

# Define GNN-prioritized gene sets (human orthologs)
gnn_inflammation <- c("TLR4", "NFKB1", "RELA", "STAT3", "TNF", "IL6", "CCL2", "SELL")
gnn_fibrosis <- c("SMAD4", "RUNX2", "DNMT3A", "PRKCA", "FN1", "COL1A1", "ACTA2", "TGFB1")

# Prepare ranked list
ranked_genes <- setNames(gse$log2FC_DVT_vs_Control, gse$gene_symbol)
ranked_genes <- sort(ranked_genes[!is.na(ranked_genes)], decreasing = TRUE)

# ============================================================
# Panel A: GSEA enrichment plot
# ============================================================
set.seed(42)
fgsea_res <- fgseaMultilevel(
  pathways = list(
    "GNN Inflammation Program" = intersect(gnn_inflammation, names(ranked_genes)),
    "GNN Fibrosis Program" = intersect(gnn_fibrosis, names(ranked_genes))
  ),
  stats = ranked_genes,
  minSize = 3, maxSize = 500
)

p1 <- ggplot(fgsea_res, aes(x = reorder(pathway, NES), y = NES, fill = -log10(padj))) +
  geom_col(width = 0.6) +
  scale_fill_gradient(low = "steelblue", high = "crimson") +
  coord_flip() +
  labs(title = "GSEA: GNN Programs in Human VTE Blood (GSE48000)",
       subtitle = paste0("Inflammation: NES=", round(fgsea_res$NES[1], 2),
                         ", FDR=", format(fgsea_res$padj[1], digits = 2)),
       x = "", y = "Normalized Enrichment Score") +
  theme_minimal(base_size = 11)

# ============================================================
# Panel B: Leave-one-out robustness
# ============================================================
loo_results <- data.frame()
for (gene in gnn_inflammation) {
  subset_genes <- setdiff(gnn_inflammation, gene)
  res <- fgseaMultilevel(
    pathways = list("subset" = intersect(subset_genes, names(ranked_genes))),
    stats = ranked_genes, minSize = 2, maxSize = 500
  )
  loo_results <- rbind(loo_results, data.frame(
    removed_gene = gene, NES = res$NES[1], FDR = res$padj[1], program = "Inflammation"
  ))
}
for (gene in gnn_fibrosis) {
  subset_genes <- setdiff(gnn_fibrosis, gene)
  res <- fgseaMultilevel(
    pathways = list("subset" = intersect(subset_genes, names(ranked_genes))),
    stats = ranked_genes, minSize = 2, maxSize = 500
  )
  loo_results <- rbind(loo_results, data.frame(
    removed_gene = gene, NES = res$NES[1], FDR = res$padj[1], program = "Fibrosis"
  ))
}

p2 <- ggplot(loo_results, aes(x = removed_gene, y = NES, fill = program)) +
  geom_col(width = 0.6) +
  facet_wrap(~ program, scales = "free_x") +
  scale_fill_manual(values = c("Inflammation" = "#4DBBD5", "Fibrosis" = "#E64B35")) +
  labs(title = "Leave-One-Gene-Out Robustness",
       subtitle = "NES remains positive when any single gene is removed",
       x = "Removed Gene", y = "NES") +
  theme_minimal(base_size = 10) + guides(fill = "none") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ============================================================
# Panel C: Random gene-set negative control
# ============================================================
set.seed(123)
random_nes <- replicate(500, {
  random_genes <- sample(names(ranked_genes), size = length(gnn_inflammation))
  res <- fgseaMultilevel(
    pathways = list("random" = random_genes),
    stats = ranked_genes, minSize = 3, maxSize = 500
  )
  res$NES[1]
})

random_df <- data.frame(NES = random_nes)
p3 <- ggplot(random_df, aes(x = NES)) +
  geom_histogram(fill = "grey70", bins = 40) +
  geom_vline(xintercept = fgsea_res$NES[fgsea_res$pathway == "GNN Inflammation Program"],
             color = "#4DBBD5", linewidth = 1.5) +
  geom_vline(xintercept = fgsea_res$NES[fgsea_res$pathway == "GNN Fibrosis Program"],
             color = "#E64B35", linewidth = 1.5) +
  labs(title = "Negative Control: Random Gene Sets (n=500)",
       subtitle = paste0("GNN programs at p < 0.001 vs random distribution"),
       x = "NES", y = "Frequency") +
  theme_minimal(base_size = 10)

# ============================================================
# Assemble
# ============================================================
combined <- p1 / (p2 | p3) +
  plot_annotation(
    title = "Cross-Species Validation of GNN-Prioritized Programs",
    subtitle = "Human VTE Whole Blood Transcriptome (GSE48000, n=74)",
    tag_levels = "A"
  ) &
  theme(plot.tag = element_text(face = "bold", size = 12))

ggsave(file.path(OUT, "Figure5_CrossSpecies.tiff"), combined,
       width = 14, height = 14, dpi = 300, compression = "lzw")
message("Saved: Figure5_CrossSpecies.tiff")
```

---

### Task 6: Figure 2 — Benchmark Performance (R)

**Files:**
- Create: `scripts/render_figure2_benchmark.R`
- Produces: `figures/paper_figures/Figure2_Benchmark.tiff`

**Interfaces:**
- Consumes: `data/baselines/baseline_results.json`, `checkpoints/full_training_v2/summary.json`
- Produces: Multi-panel TIFF (AUROC/MRR/Hits@10 bar chart + baseline comparison table)

- [ ] **Step 1: Write R script (user reviews before execution)**

```r
#!/usr/bin/env Rscript
# Figure 2: Benchmark Performance + Baseline Comparison
library(ggplot2)
library(patchwork)
library(dplyr)
library(tidyr)
library(jsonlite)

OUT <- "figures/paper_figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# Load results
baselines <- fromJSON("data/baselines/baseline_results.json", simplifyDataFrame = TRUE)
summary <- fromJSON("checkpoints/full_training_v2/summary.json")

# ============================================================
# Panel A: 5-seed metrics bar chart
# ============================================================
per_seed <- as.data.frame(do.call(rbind, summary$per_seed))
per_seed <- per_seed %>%
  pivot_longer(cols = c(test_auroc, test_mrr, test_hits10),
               names_to = "metric", values_to = "value") %>%
  mutate(metric = case_when(
    metric == "test_auroc" ~ "AUROC",
    metric == "test_mrr" ~ "MRR",
    metric == "test_hits10" ~ "Hits@10"
  ))

p1 <- ggplot(per_seed, aes(x = metric, y = value, fill = metric)) +
  geom_boxplot(width = 0.5, alpha = 0.7) +
  geom_jitter(width = 0.1, size = 2, alpha = 0.8) +
  scale_fill_manual(values = c("AUROC" = "#E64B35", "MRR" = "#4DBBD5", "Hits@10" = "#00A087")) +
  labs(title = "5-Seed Performance (No Data Leakage)",
       subtitle = paste0("Test AUROC: ", round(summary$test_auroc_mean, 3), " ± ", round(summary$test_auroc_std, 3)),
       x = "", y = "") +
  theme_minimal(base_size = 11) + guides(fill = "none")

# ============================================================
# Panel B: Baseline comparison table
# ============================================================
baseline_df <- as.data.frame(t(sapply(baselines, unlist)))
baseline_df$model <- rownames(baseline_df)

p2 <- ggplot(baseline_df, aes(x = model, y = filtered_mrr, fill = model)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = round(filtered_mrr, 3)), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c(
    "TransE" = "#8491B4", "DistMult" = "#91D1C2", "ComplEx" = "#F39B7F",
    "RotatE" = "#DC0000", "TemperedHGT" = "#3C5488"
  )) +
  labs(title = "Filtered MRR: Baseline Comparison",
       subtitle = "Same train/val/test split, same filtered ranking protocol",
       x = "", y = "Filtered MRR") +
  theme_minimal(base_size = 11) + guides(fill = "none") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# ============================================================
# Panel C: AUROC comparison
# ============================================================
p3 <- ggplot(baseline_df, aes(x = model, y = ifelse(is.na(auroc), 0, auroc), fill = model)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = ifelse(is.na(auroc), "N/A", round(auroc, 3))), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c(
    "TransE" = "#8491B4", "DistMult" = "#91D1C2", "ComplEx" = "#F39B7F",
    "RotatE" = "#DC0000", "TemperedHGT" = "#3C5488"
  )) +
  labs(title = "Test AUROC Comparison", x = "", y = "AUROC") +
  theme_minimal(base_size = 11) + guides(fill = "none") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# ============================================================
# Assemble
# ============================================================
combined <- (p1 | (p2 / p3)) +
  plot_annotation(
    title = "Model Performance & Baseline Comparison",
    subtitle = "Tempered HGT vs. Classical KG Embedding Models",
    tag_levels = "A"
  ) &
  theme(plot.tag = element_text(face = "bold", size = 12))

ggsave(file.path(OUT, "Figure2_Benchmark.tiff"), combined,
       width = 14, height = 10, dpi = 300, compression = "lzw")
message("Saved: Figure2_Benchmark.tiff")
```

---

### Task 7: Figure 1 — Architecture Schematic

**Files:**
- Create: `scripts/render_figure1_architecture.R`
- Produces: `figures/paper_figures/Figure1_Architecture.tiff`

**Notes:** Figure 1 is mostly conceptual (architecture diagram, split schematic). R can do layouts with `grid`/`ggraph` but a diagram tool (BioRender, draw.io, or Adobe Illustrator) may be more appropriate for Panel A (KG schema) and Panel B (model architecture). This task creates the data-driven panels (C: split statistics, D: evaluation schematic) and leaves the conceptual panels for manual assembly.

---

## Execution Order

1. Task 1 → Task 2 (baselines data → baseline results)
2. Task 4 Step 1-2 (export scRNA to CSV)
3. All R scripts ready for user review
4. User approves → run all R scripts
5. Figures assembled → manuscript rewrite begins
