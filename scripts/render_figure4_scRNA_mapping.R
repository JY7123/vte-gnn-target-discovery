#!/usr/bin/env Rscript
# Figure 4: scRNA-seq Cell-Type Mapping of GNN-Prioritized Network Genes (Refined)
library(ggplot2)
library(patchwork)
library(dplyr)
library(tidyr)
library(scales)

if (requireNamespace("rstudioapi", quietly = TRUE) &&
    rstudioapi::isAvailable()) {
  setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
}
OUT <- "figures/paper_figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# Load
# ============================================================
expr_df <- read.csv("figures/scRNA/gnn_network_expression.csv")
umap_df <- read.csv("figures/scRNA/umap_coords.csv")

# ============================================================
# Panel A & B: UMAP
# ============================================================
p1a <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = condition)) +
  geom_point(size = 0.2, alpha = 0.6) +
  scale_color_manual(values = c("Control" = "steelblue", "DVT" = "#DC0000")) +
  labs(title = "A  Condition", x = "UMAP1", y = "UMAP2") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold", size = 12)) +
  guides(color = guide_legend(override.aes = list(size = 3)))

cell_colors <- c(
  "Endothelial" = "#E64B35", "Fibroblast" = "#4DBBD5", "VSMC" = "#00A087",
  "Macrophage" = "#3C5488", "Monocyte" = "#F39B7F", "Neutrophil" = "#8491B4",
  "B_cell" = "#91D1C2", "Erythrocyte" = "#DC0000"
)

p1b <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = cell_type)) +
  geom_point(size = 0.2, alpha = 0.6) +
  scale_color_manual(values = cell_colors) +
  labs(title = "B  Cell Type", x = "UMAP1", y = "UMAP2") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold", size = 12)) +
  guides(color = guide_legend(override.aes = list(size = 3), ncol = 2))

# ============================================================
# Panel C: Dotplot — Fixed Gene Ordering
# ============================================================
infl_genes <- c("Tlr4", "Nfkb1", "Rela", "Stat3", "Spp1", "Tnf", "Il6", "Ccl2", "Sell")
fib_genes  <- c("Smad4", "Runx2", "Dnmt3a", "Prkca", "Fn1", "Col1a1", "Acta2")
all_ordered_genes <- c(infl_genes, fib_genes, "Tgfb1")

dvt_expr <- expr_df %>%
  filter(condition == "DVT") %>%
  filter(gene %in% all_ordered_genes) %>%
  mutate(
    gene = factor(gene, levels = all_ordered_genes),
    gene_program = case_when(
      gene %in% infl_genes ~ "Inflammation Axis",
      gene %in% fib_genes  ~ "Fibrosis Axis",
      gene == "Tgfb1"      ~ "TGFB Ligand"
    ),
    gene_program = factor(gene_program, levels = c("Inflammation Axis", "Fibrosis Axis", "TGFB Ligand"))
  )

p2 <- ggplot(dvt_expr, aes(x = gene, y = cell_type)) +
  geom_point(aes(size = pct_positive, color = mean_expr)) +
  scale_color_gradient2(low = "steelblue", mid = "white", high = "#DC0000",
                        midpoint = median(dvt_expr$mean_expr)) +
  scale_size_continuous(range = c(1, 8), name = "% Positive") +
  facet_grid(. ~ gene_program, scales = "free_x", space = "free_x") +
  labs(title = "C  GNN-Prioritized Network Genes: Cell-Type Expression in DVT Vein Wall",
       subtitle = "Dot size = % positive cells; Color = mean log-normalized expression",
       x = "", y = "") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, face = "italic"),
        strip.text = element_text(face = "bold", size = 10),
        plot.title = element_text(face = "bold", size = 12))

# ============================================================
# Panel D: TGFB1 → Fibrosis Axis (Macrophage Ligand → Fibroblast Effectors)
# ============================================================
tgf_data <- expr_df %>%
  filter(gene %in% c("Tgfb1", "Smad4", "Col1a1", "Fn1")) %>%
  filter(cell_type %in% c("Macrophage", "Fibroblast")) %>%
  mutate(gene = factor(gene, levels = c("Tgfb1", "Smad4", "Col1a1", "Fn1")))

p3 <- ggplot(tgf_data, aes(x = cell_type, y = mean_expr, fill = condition)) +
  geom_col(position = position_dodge(0.8), width = 0.6) +
  geom_text(aes(label = sprintf("%.2f", mean_expr), group = condition),
            position = position_dodge(0.8), vjust = -0.3, size = 3) +
  facet_wrap(~ gene, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = c("Control" = "steelblue", "DVT" = "#DC0000")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(title = "D  TGF-β → Fibrosis Axis: Macrophage-Fibroblast Crosstalk",
       subtitle = "Tgfb1 (ligand, Macrophage) → Smad4/Col1a1/Fn1 (effectors, Fibroblast)",
       x = "", y = "Mean Expression") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        plot.title = element_text(face = "bold", size = 12),
        strip.text = element_text(face = "italic"))

# ============================================================
# Panel E: TLR4/NF-κB Inflammation Program
# ============================================================
infl_data <- expr_df %>%
  filter(gene %in% c("Tlr4", "Nfkb1", "Spp1", "Tnf")) %>%
  filter(cell_type %in% c("Macrophage", "Monocyte", "Neutrophil")) %>%
  mutate(gene = factor(gene, levels = c("Tlr4", "Nfkb1", "Spp1", "Tnf")))

p4 <- ggplot(infl_data, aes(x = cell_type, y = mean_expr, fill = condition)) +
  geom_col(position = position_dodge(0.8), width = 0.6) +
  geom_text(aes(label = sprintf("%.2f", mean_expr), group = condition),
            position = position_dodge(0.8), vjust = -0.3, size = 3) +
  facet_wrap(~ gene, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = c("Control" = "steelblue", "DVT" = "#DC0000")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(title = "E  TLR4/NF-κB Inflammation Program in Myeloid Cells",
       x = "", y = "Mean Expression") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        plot.title = element_text(face = "bold", size = 12),
        strip.text = element_text(face = "italic"))

# ============================================================
# Assemble
# ============================================================
top_row <- p1a + p1b + plot_layout(widths = c(1, 1.3))
combined <- (top_row / p2 / p3 / p4) +
  plot_annotation(
    title = "Figure 4: Single-Cell Mapping of GNN-Prioritized Pathological Programs",
    subtitle = paste0("Mouse IVC Stenosis Model, Day 14 | ",
                      nrow(umap_df), " cells, ",
                      length(unique(expr_df$cell_type)), " cell types | ",
                      "IVC vein wall tissue")
  ) &
  theme(plot.title = element_text(face = "bold", size = 15))

ggsave(file.path(OUT, "Figure4_scRNA_Mapping.tiff"), combined,
       width = 16, height = 18, dpi = 300, compression = "lzw")
message("Saved: Figure4_scRNA_Mapping.tiff")

