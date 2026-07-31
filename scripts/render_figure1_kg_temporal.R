#!/usr/bin/env Rscript
# Figure 1: Knowledge Graph Construction & Strict Temporal Split Framework
# Data aligned with Methods: 82,644 nodes, 29 curated relations, 11,989 train edges
library(ggplot2)
library(patchwork)
library(dplyr)
library(scales)

OUT <- "figures/paper_figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# Panel a: Node Type Distribution (14 types, 82,644 total nodes)
# ============================================================
nodes_df <- data.frame(
  Type = c("Article", "Disease", "Protein", "Cell", "Gene", "Process",
           "Metabolite", "Pathway", "Drug", "Cytokine", "Concept", "ECM", "Hormone"),
  Count = c(20869, 14597, 14283, 11625, 4959, 4723, 3664, 2251, 2136, 1511, 1355, 458, 213)
)
nodes_df$Type <- factor(nodes_df$Type, levels = nodes_df$Type[order(nodes_df$Count)])

p1a <- ggplot(nodes_df, aes(x = Type, y = Count)) +
  geom_col(fill = "#008B8B", width = 0.7) +
  geom_text(aes(label = comma(Count)), hjust = -0.15, size = 3) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 26000), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "a  Node Type Distribution",
       subtitle = "82,644 nodes across 14 entity types",
       x = "", y = "Number of Nodes") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold", size = 12))

# ============================================================
# Panel b: Curated Edge Type Statistics (29 relation types, 11,989 edges)
# ============================================================
edges_df <- data.frame(
  Category = c("Gene/Protein Regulation", "Disease Association", "Drug-Target",
               "Pathway/Process", "Cytokine/Cell Signaling"),
  Relation_Count = c(12, 7, 2, 4, 4),
  Edge_Count = c(5639, 5542, 1178, 168, 1462)
)
edges_df$Category <- factor(edges_df$Category, levels = edges_df$Category[order(edges_df$Edge_Count)])

p1b <- ggplot(edges_df, aes(x = Category, y = Edge_Count)) +
  geom_col(fill = "#3C5488", width = 0.55) +
  geom_text(aes(label = paste0(comma(Edge_Count), " edges\n(", Relation_Count, " relations)")),
            vjust = -0.2, size = 3, lineheight = 0.85) +
  scale_y_continuous(limits = c(0, 7200), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "b  Curated Edge Type Distribution",
       subtitle = "11,989 edges across 29 biomedical relation types",
       x = "", y = "Number of Edges") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold", size = 12))

# ============================================================
# Panel c: Strict Temporal Split (factor levels enforce chronological order)
# ============================================================
split_levels <- c("Train\n(≤ 2024)", "Validation\n(2025 H1)", "Test\n(2025 H2 – 2026 H1)")
temporal_df <- data.frame(
  Split = factor(split_levels, levels = split_levels),
  Edges = c(11989, 850, 1150)
)

p1c <- ggplot(temporal_df, aes(x = Split, y = Edges, fill = Split)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = paste0(comma(Edges), " edges")),
            vjust = -0.4, fontface = "bold", size = 3.8) +
  scale_fill_manual(values = c("Train\n(≤ 2024)" = "#4DBBD5",
                               "Validation\n(2025 H1)" = "#E64B35",
                               "Test\n(2025 H2 – 2026 H1)" = "#00A087")) +
  scale_y_continuous(limits = c(0, 15000), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "c  Strict Temporal Split (No Data Leakage)",
       subtitle = "Edges partitioned by publication date; 5 independent random seeds",
       x = "", y = "Number of Edges") +
  theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold", size = 12),
        legend.position = "none")

# ============================================================
# Assemble a-b-c without d (d = existing architecture schematic,
# cropped from Figure1_KG_Architecture.png and pasted in PPT)
# ============================================================
combined <- (p1a | p1b) | p1c +
  plot_annotation(
    title = "Figure 1: Knowledge Graph Construction and Temporal Evaluation Framework",
    subtitle = "Curated heterogeneous KG with strict leakage-free train/test partitioning"
  ) &
  theme(plot.title = element_text(face = "bold", size = 14))

# Export individual panels for manual assembly with architecture diagram (Panel d)
# Panel d is reused from the original Figure1_KG_Architecture.png (model unchanged)
ggsave(file.path(OUT, "Figure1_Panel_a.tiff"), p1a, width = 6, height = 5, dpi = 300, compression = "lzw")
ggsave(file.path(OUT, "Figure1_Panel_b.tiff"), p1b, width = 6, height = 5, dpi = 300, compression = "lzw")
ggsave(file.path(OUT, "Figure1_Panel_c.tiff"), p1c, width = 6, height = 5, dpi = 300, compression = "lzw")

message("Saved: Panel a/b/c to ", OUT)
message("Panel d: crop architecture diagram from the existing Figure1_KG_Architecture.png")
message("Assemble all 4 panels in PPT/Keynote → 2x2 layout, export 300 DPI TIFF")
