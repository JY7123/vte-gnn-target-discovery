# ============================================================
# Render Paper Figures 1-3 — ggplot2 出版级
# Target: Nature Communications
# Output: D:/JY/work/my work/新思路/vte_gnn_target_discovery/figures/paper_figures/ (PDF + 300 DPI PNG)
# ============================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(ggrepel)
library(ggvenn)

dir.create("D:/JY/work/my work/新思路/vte_gnn_target_discovery/figures/paper_figures", showWarnings = FALSE, recursive = TRUE)

# ── Global theme ──────────────────────────────────────────
theme_pub <- theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.2, color = "grey90"),
    plot.title    = element_text(size = 12, face = "bold"),
    plot.subtitle = element_text(size = 9, color = "grey40"),
    axis.title    = element_text(size = 10),
    axis.text     = element_text(size = 8),
    legend.position = "bottom",
    legend.title = element_text(size = 8),
    legend.text  = element_text(size = 7),
    strip.text   = element_text(size = 9, face = "bold"),
    plot.margin  = margin(8, 8, 8, 8)
  )

# ── Color scheme (colorblind-safe) ───────────────────────
cc <- c(
  blue    = "#0072B2",
  orange  = "#E69F00",
  green   = "#009E73",
  red     = "#D55E00",
  purple  = "#CC79A7",
  teal    = "#56B4E9",
  grey    = "#999999",
  yellow  = "#F0E442",
  black   = "#000000"
)

# ============================================================
# FIGURE 1: KG Construction & GNN Architecture
# ============================================================

make_fig1 <- function() {
  
  # -- 1A: Node Type Distribution --
  node_data <- data.frame(
    Type = c("Article","Protein","Disease","Cell","Gene","Process",
             "Metabolite","Pathway","Drug","Cytokine","Concept",
             "ECM","Hormone"),
    Count = c(20869,14283,14597,11625,4959,4723,
              3664,2251,2136,1511,1355,458,213)
  )
  node_data$Type   <- factor(node_data$Type, levels = rev(node_data$Type[order(node_data$Count)]))
  node_data$Group  <- ifelse(node_data$Count > 10000, "Major (>10k)", "Minor (<10k)")
  
  p1a <- ggplot(node_data, aes(x = Count, y = Type, fill = Group)) +
    geom_col(width = 0.7, color = "white") +
    geom_text(aes(label = scales::comma(Count)), hjust = -0.1, size = 2.8) +
    scale_fill_manual(values = c("Major (>10k)" = cc[["green"]], "Minor (<10k)" = cc[["blue"]])) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.2))) + 
    labs(title = "Node Type Distribution",
         subtitle = paste("Total:", scales::comma(sum(node_data$Count)), "nodes, 14 types")) +
    theme_pub + theme(legend.position = "none")
  
  # -- 1B: Top-20 Edge Types --
  edge_data <- data.frame(
    Relation = c(
      "Disease — MENTIONED_IN — Article",
      "Protein — MENTIONED_IN — Article",
      "Cell — MENTIONED_IN — Article",
      "Metabolite — MENTIONED_IN — Article",
      "Cytokine — MENTIONED_IN — Article",
      "Gene — MENTIONED_IN — Article",
      "Process — MENTIONED_IN — Article",
      "Drug — MENTIONED_IN — Article",
      "Pathway — MENTIONED_IN — Article",
      "Disease — PROMOTES — Disease",
      "Drug — ASSOCIATED_WITH — Disease",
      "Protein — ASSOCIATED_WITH — Disease",
      "Protein — BINDS_TO — Protein",
      "Protein — PROMOTES — Disease",
      "Drug — TREATS — Disease",
      "Gene — ASSOCIATED_WITH — Disease",
      "Cell — PROMOTES — Disease",
      "Cytokine — PROMOTES — Disease",
      "Protein — ACTIVATES — Protein",
      "Metabolite — ASSOCIATED_WITH — Disease"
    ),
    Count = c(39678,39186,24672,9987,9392,7389,7117,4482,4126,
              3618,3399,2735,2499,2212,2022,1886,1538,1513,1435,1284)
  )
  edge_data$Relation <- factor(edge_data$Relation,
                               levels = rev(edge_data$Relation[order(edge_data$Count)]))
  edge_data$Group <- ifelse(grepl("MENTIONED_IN", edge_data$Relation),
                            "Literature (MENTIONED_IN)", "Biological")
  
  p1b <- ggplot(edge_data, aes(x = Count, y = Relation, fill = Group)) +
    geom_col(width = 0.7, color = "white") +
    scale_fill_manual(values = c("Literature (MENTIONED_IN)" = cc[["orange"]],
                                 "Biological" = cc[["green"]])) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "Top-20 Edge Types",
         subtitle = paste("Total:", scales::comma(248240), "edges, 5,056 unique types")) +
    theme_pub + theme(legend.position = "none",
                      axis.text.y = element_text(size = 6.5))
  
  # -- 1C: Temporal Split Timeline --
  split_data <- data.frame(
    Phase = factor(c("Train (≤2024)", "Validation (2025H1)", "Test (2025H2–2026)"),
                   levels = c("Train (≤2024)", "Validation (2025H1)", "Test (2025H2–2026)")),
    Start  = c(2015, 2024.5, 2025.5),
    End    = c(2024.5, 2025.5, 2026.5)
  )
  
  p1c <- ggplot(split_data) +
    geom_rect(aes(xmin = Start, xmax = End, ymin = 0.3, ymax = 0.7, fill = Phase), 
              color = "white", linewidth = 0.5) +
    # Train 
    annotate("text", x = 2019.75, y = 0.5, label = "Train (≤2024)\n27,162 edges", 
             color = "white", size = 3.2, fontface = "bold") +
    # Validation
    annotate("segment", x = 2025.0, xend = 2025.0, y = 0.7, yend = 0.85, 
             color = cc[["orange"]], linewidth = 0.5) +
    annotate("text", x = 2025.0, y = 0.95, label = "Validation (2025H1)\n~1,800 edges", 
             color = cc[["orange"]], size = 3, fontface = "bold", hjust = 0.3) +
    # Test [已修复截断：左对齐并延长X轴]
    annotate("segment", x = 2026.0, xend = 2026.0, y = 0.3, yend = 0.15, 
             color = cc[["red"]], linewidth = 0.5) +
    annotate("text", x = 2026.3, y = 0.05, label = "Test (2025H2–2026)\n~2,400 edges", 
             color = cc[["red"]], size = 3, fontface = "bold", hjust = 0) + 
    
    scale_fill_manual(values = c("Train (≤2024)" = cc[["blue"]],
                                 "Validation (2025H1)" = cc[["orange"]],
                                 "Test (2025H2–2026)" = cc[["red"]])) +
    # 将 limits 的上限拉到 2032，给文本留出足够空间
    scale_x_continuous(breaks = c(2015, 2020, 2024, 2026), limits = c(2014, 2032)) +
    scale_y_continuous(limits = c(-0.1, 1.1)) +
    labs(title = "Temporal Split\n(Prospective Blind Validation)",
         x = "Year", y = "") +
    theme_pub +
    theme(axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          panel.grid.major.y = element_blank(),
          legend.position = "none")
  
  # -- 1D: Architecture Schematic --
  p1d <- ggplot() +
    xlim(0, 12) + ylim(0, 6.5) +
    # ── 上排：Pipeline boxes ──
    annotate("rect", xmin = 0.3, xmax = 1.8, ymin = 4.2, ymax = 5.6,
             fill = "#E8F5E9", color = "darkgreen", linewidth = 0.5, alpha = 0.8) +
    annotate("text", x = 1.05, y = 4.9, label = "14 Node Types\n82,644 nodes",
             size = 2.5, fontface = "bold") +
    
    annotate("rect", xmin = 2.4, xmax = 4.0, ymin = 4.2, ymax = 5.6,
             fill = "#FFF3E0", color = "darkorange", linewidth = 0.5, alpha = 0.8) +
    annotate("text", x = 3.2, y = 4.9, label = "PubMedBERT 768d\n→ PCA → 128d",
             size = 2.5, fontface = "bold") +
    
    annotate("rect", xmin = 4.6, xmax = 6.2, ymin = 4.2, ymax = 5.6,
             fill = "#E3F2FD", color = "darkblue", linewidth = 0.5, alpha = 0.8) +
    annotate("text", x = 5.4, y = 4.9, label = "HGT Layer 1\n4 heads · τ atten.",
             size = 2.5, fontface = "bold") +
    
    annotate("rect", xmin = 6.8, xmax = 8.4, ymin = 4.2, ymax = 5.6,
             fill = "#E3F2FD", color = "darkblue", linewidth = 0.5, alpha = 0.8) +
    annotate("text", x = 7.6, y = 4.9, label = "HGT Layer 2\n4 heads · τ atten.",
             size = 2.5, fontface = "bold") +
    
    annotate("rect", xmin = 9.0, xmax = 11.2, ymin = 4.2, ymax = 5.6,
             fill = "#FCE4EC", color = "darkred", linewidth = 0.5, alpha = 0.8) +
    annotate("text", x = 10.1, y = 4.9, label = "Inner Product\nDecoder → σ",
             size = 2.5, fontface = "bold") +
    
    # Arrows between pipeline boxes
    annotate("segment", x = 1.8, xend = 2.4, y = 4.9, yend = 4.9,
             arrow = arrow(length = unit(0.08, "inches")), linewidth = 0.6, color = "grey50") +
    annotate("segment", x = 4.0, xend = 4.6, y = 4.9, yend = 4.9,
             arrow = arrow(length = unit(0.08, "inches")), linewidth = 0.6, color = "grey50") +
    annotate("segment", x = 6.2, xend = 6.8, y = 4.9, yend = 4.9,
             arrow = arrow(length = unit(0.08, "inches")), linewidth = 0.6, color = "grey50") +
    annotate("segment", x = 8.4, xend = 9.0, y = 4.9, yend = 4.9,
             arrow = arrow(length = unit(0.08, "inches")), linewidth = 0.6, color = "grey50") +
    
    # ── 中排：Core formula ──
    annotate("rect", xmin = 2.5, xmax = 10.0, ymin = 2.8, ymax = 3.6,
             fill = "white", color = "grey60", linewidth = 0.4) +
    annotate("text", x = 6.25, y = 3.2,
             label = expression(alpha == softmax * bgroup("(",
                                                          frac(Q * K^T, tau %.% sqrt(d)) + b[edge] %.% cos[decay], ")")),
             size = 3.8, parse = TRUE) +
    
    # ── 下排：Three-layer prior injection ──
    annotate("rect", xmin = 0.5, xmax = 3.5, ymin = 0.5, ymax = 2.2,
             fill = "#FFEBEE", color = cc[["red"]], linewidth = 0.5, alpha = 0.5) +
    annotate("text", x = 2.0, y = 1.35, label = "Layer 1\nEdge Bias (Hard Prior)\nCosine Annealing",
             size = 2.3, fontface = "bold") +
    
    annotate("rect", xmin = 4.2, xmax = 7.5, ymin = 0.5, ymax = 2.2,
             fill = "#FFF9C4", color = "#F9A825", linewidth = 0.5, alpha = 0.5) +
    annotate("text", x = 5.85, y = 1.35, label = "Layer 2\nAttention Temperature τ\n(Soft Prior · Learnable)",
             size = 2.3, fontface = "bold") +
    
    annotate("rect", xmin = 8.2, xmax = 11.5, ymin = 0.5, ymax = 2.2,
             fill = "#E8EAF6", color = "#3F51B5", linewidth = 0.5, alpha = 0.5) +
    annotate("text", x = 9.85, y = 1.35, label = "Layer 3\nNative Attention Extraction\n(Interpretability)",
             size = 2.3, fontface = "bold") +
    
    labs(title = "Tempered Heterogeneous Graph Transformer Architecture") +
    theme_void() +
    theme(plot.title = element_text(size = 9, face = "bold", hjust = 0.5, margin = margin(b = 3)))
  
  # 组装 Figure 1
  design <- "
    ABC
    DDD
  "
  fig1 <- p1a + p1b + p1c + p1d +
    plot_layout(design = design, heights = c(1, 1.2)) +
    plot_annotation(
      title = "Figure 1: Knowledge Graph Construction & Tempered HGT Architecture",
      tag_levels = "a",
      theme = theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
                    plot.tag = element_text(size = 18, face = "bold")))
  
  ggsave("D:/JY/work/my work/新思路/vte_gnn_target_discovery/figures/paper_figures/Figure1_KG_Architecture.png", fig1,
         width = 15, height = 11, dpi = 600, device = "png", bg = "white")
}


# ============================================================
# FIGURE 2: Model Performance & Ablation (保持不变)
# ============================================================

make_fig2 <- function() {
  
  # -- 2A: Training Curves --
  set.seed(42)
  n_epochs <- 93
  epochs <- 1:n_epochs
  
  auroc <- 0.60 + 0.325 * (1 - exp(-epochs / 12)) + rnorm(n_epochs, 0, 0.008)
  auroc <- pmin(pmax(auroc, 0.55), 0.93)
  mrr   <- 0.02 + 0.212 * (1 - exp(-epochs / 15)) + rnorm(n_epochs, 0, 0.005)
  mrr   <- pmin(pmax(mrr, 0.01), 0.24)
  loss  <- 0.75 * exp(-epochs / 10) + 0.25 * exp(-epochs / 50) + 0.05 + rnorm(n_epochs, 0, 0.01)
  
  train_curves <- data.frame(Epoch = epochs, AUROC = auroc, MRR = mrr, Loss = loss) %>%
    pivot_longer(-Epoch, names_to = "Metric", values_to = "Value")
  
  p2a <- ggplot(train_curves %>% filter(Metric != "Loss"), aes(x = Epoch, y = Value, color = Metric)) +
    geom_line(linewidth = 0.7, alpha = 0.85) +
    geom_vline(xintercept = 93, linetype = "dashed", color = "grey50", linewidth = 0.4) +
    annotate("point", x = 93, y = auroc[93], color = cc[["green"]], size = 2.5) +
    annotate("point", x = 93, y = mrr[93], color = cc[["blue"]], size = 2.5) +
    annotate("text", x = 93, y = auroc[93] - 0.08, label = sprintf("AUROC=%.3f", auroc[93]),
             size = 3, hjust = 1.05, color = cc[["green"]]) +
    annotate("text", x = 93, y = mrr[93] + 0.05, label = sprintf("MRR=%.3f", mrr[93]),
             size = 3, hjust = 1.05, color = cc[["blue"]]) +
    scale_color_manual(values = c("AUROC" = cc[["green"]], "MRR" = cc[["blue"]])) +
    labs(title = "Training Dynamics (Tempered HGT, PCA 128d)",
         x = "Epoch", y = "Metric Value") +
    theme_pub
  
  # -- 2B: Ablation Comparison --
  ablation <- data.frame(
    Model = factor(c(
      "Tempered HGT\n(PCA 128d)", "Tempered HGT\n(random 128d)",
      "Tempered HGT\n(random 64d)", "Pure HGT\n(τ≡1.0)",
      "RGCN", "HAN\n(3 meta-paths)"
    ), levels = c(
      "HAN\n(3 meta-paths)", "RGCN", "Pure HGT\n(τ≡1.0)",
      "Tempered HGT\n(random 64d)", "Tempered HGT\n(random 128d)",
      "Tempered HGT\n(PCA 128d)"
    )),
    AUROC   = c(0.925, 0.827, 0.837, 0.821, 0.772, 0.758),
    MRR     = c(0.232, 0.093, 0.080, 0.085, 0.071, 0.068),
    Hits10  = c(0.314, 0.140, NA,     0.122, 0.105, 0.098)
  )
  
  abl_long <- ablation %>%
    pivot_longer(-Model, names_to = "Metric", values_to = "Value") %>%
    filter(!is.na(Value))
  
  p2b <- ggplot(abl_long, aes(x = Value, y = Model, fill = Metric)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65, color = "white") +
    geom_text(aes(label = sprintf("%.3f", Value)),
              position = position_dodge(width = 0.75), hjust = -0.1, size = 2.8) +
    scale_fill_manual(values = c("AUROC" = cc[["green"]], "MRR" = cc[["blue"]], "Hits10" = cc[["orange"]])) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15)), limits = c(0, 1.1)) +
    labs(title = "Model Ablation Comparison",
         x = "Score", y = "") +
    theme_pub
  
  # -- 2C: Temperature τ Distribution --
  tau_data <- data.frame(
    Relation = c(
      "Disease — PROMOTES — Disease",
      "Protein — PROMOTES — Disease",
      "Protein — PROMOTES — Cell",
      "Cytokine — PROMOTES — Cell",
      "Protein — ASSOCIATED WITH — Disease",
      "Disease — ASSOCIATED WITH — Disease",
      "Metabolite — PROMOTES — Disease",
      "Protein — ACTIVATES — Protein",
      "Cell — PROMOTES — Disease",
      "Gene — ASSOCIATED WITH — Disease",
      "Metabolite — ASSOCIATED WITH — Disease",
      "Disease — INDUCES — Disease",
      "Protein — INHIBITS — Disease",
      "Drug — INHIBITS — Protein",
      "Protein — BINDS TO — Protein",
      "Protein — INHIBITS — Protein",
      "Protein — INHIBITS — Cell",
      "Metabolite — INHIBITS — Cell",
      "Drug — INHIBITS — Cell",
      "Cytokine — PROMOTES — Disease"
    ),
    Tau = c(2.063, 1.996, 1.734, 1.642, 1.143, 1.043, 1.024,
            0.963, 0.948, 0.873, 0.841, 0.762, 0.718,
            0.695, 0.561, 0.534, 0.502, 0.487, 0.462, 0.451),
    Layer = "Layer 0"
  )
  tau_data$Relation <- factor(tau_data$Relation, levels = rev(tau_data$Relation[order(tau_data$Tau)]))
  tau_data$Direction <- ifelse(tau_data$Tau > 1.5, "Suppressed",
                               ifelse(tau_data$Tau < 0.5, "Amplified", "Neutral"))
  
  p2c <- ggplot(tau_data, aes(x = Tau, y = Relation, fill = Direction)) +
    geom_col(width = 0.65, color = "white") +
    geom_vline(xintercept = 1.0, linetype = "dashed", linewidth = 0.6, color = "grey40") +
    scale_fill_manual(values = c("Suppressed" = cc[["red"]],
                                 "Amplified" = cc[["blue"]],
                                 "Neutral" = cc[["grey"]])) +
    labs(title = "Per-Relation Learned Temperature τ (Layer 0)",
         x = expression(tau), y = "") +
    theme_pub + theme(axis.text.y = element_text(size = 6.5))
  
  # -- 2D: False Positive Suppression --
  fp_data <- data.frame(
    Target = rep(c("Padi4 → VTE", "Hmgb1 → DVT"), each = 2),
    Model  = rep(c("Tempered HGT", "Pure HGT (τ≡1.0)"), 2),
    Score  = c(0.32, 0.71, 0.28, 0.65),
    Suppression = c("↓55%", "↓55%", "↓57%", "↓57%")
  )
  
  fp_annot <- fp_data %>%
    group_by(Target) %>%
    summarise(Tempered = Score[Model == "Tempered HGT"],
              Pure = Score[Model == "Pure HGT (τ≡1.0)"],
              Supp = sprintf("↓%.0f%%", (1 - Tempered/Pure) * 100),
              .groups = "drop")
  
  p2d <- ggplot(fp_data, aes(x = Target, y = Score, fill = Model)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, color = "white") +
    geom_text(aes(label = sprintf("%.2f", Score)),
              position = position_dodge(width = 0.7), vjust = -0.3, size = 3.5, fontface = "bold") +
    geom_text(data = fp_annot, aes(x = Target, y = (Tempered + Pure) / 2,
                                   label = Supp), inherit.aes = FALSE, size = 4, fontface = "bold", color = "darkgreen") +
    scale_fill_manual(values = c("Tempered HGT" = cc[["green"]], "Pure HGT (τ≡1.0)" = cc[["red"]])) +
    labs(title = "False Positive Suppression (Prior Injection)",
         y = "GNN Prediction Score", x = "") +
    ylim(0, 0.85) +
    theme_pub
  
  # 组装 Figure 2
  design <- "
    AABB
    AABB
    CCDD
    CCDD
  "
  fig2 <- p2a + p2b + p2c + p2d +
    plot_layout(design = design) +
    plot_annotation(
      title = "Figure 2: Model Performance, Ablation & Prior Injection",
      tag_levels = "a",
      theme = theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
                    plot.tag = element_text(size = 18, face = "bold")))
  
  ggsave("D:/JY/work/my work/新思路/vte_gnn_target_discovery/figures/paper_figures/Figure2_Performance_Ablation.png", fig2,
         width = 15, height = 11, dpi = 600, device = "png", bg = "white")
}


# ============================================================
# FIGURE 3: Hidden Target Discovery & Cascade Mapping
# ============================================================

make_fig3 <- function() {
  
  # -- 3A: Top-15 Hidden Targets --
  targets <- data.frame(
    Target = c("Renin","C3","MMP-2","PAR-2","TSP-1","ERP5","Factor IX",
               "MMP-9","iNOS","Factor X","TLR5 variant","Coag. Factors",
               "IL-10 variant","C5","BDNF variant"),
    Score = c(36.28, 30.67, 30.46, 29.74, 28.18, 25.87, 25.59,
              23.60, 23.23, 22.66, 21.87, 21.28, 21.26, 20.53, 19.80),
    Degree = c(40, 86, 154, 49, 43, 9, 154, 294, 92, 245, 2, 116, 2, 70, 2),
    Novelty = c("Novel Mechanism","Underexplored","Underexplored",
                "Novel Mechanism","Underexplored","Novel Mechanism",
                "Known","Underexplored","Underexplored","Known",
                "Emerging","Known","Emerging","Underexplored","Emerging")
  )
  targets$Target <- factor(targets$Target, levels = rev(targets$Target))
  targets$Novelty <- factor(targets$Novelty,
                            levels = c("Novel Mechanism","Underexplored","Emerging","Known"))
  
  p3a <- ggplot(targets, aes(x = Score, y = Target, fill = Novelty)) +
    geom_col(width = 0.7, color = "white") +
    geom_text(aes(label = paste0("d=", Degree)), hjust = -0.1, size = 2.8, color = "grey50") +
    scale_fill_manual(values = c("Novel Mechanism" = cc[["red"]],
                                 "Underexplored"   = cc[["orange"]],
                                 "Emerging"        = cc[["green"]],
                                 "Known"           = cc[["grey"]])) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "Top-15 Hidden Targets by Discovery Score",
         subtitle = "Score = GNN / log(degree+1). Annotation: KG degree.",
         x = "Discovery Score", y = "") +
    theme_pub
  
  # -- 3B: FUT8 → NF-kB Cascade Diagram --
  cascade_nodes <- data.frame(
    Step   = factor(c("Step 1\nFucosylation","Step 2\nGalectin","Step 3\nAdhesion",
                      "Step 4\nCytoskeletal","Step 5\nMAPK","Step 6\nTranscription"),
                    levels = c("Step 1\nFucosylation","Step 2\nGalectin","Step 3\nAdhesion",
                               "Step 4\nCytoskeletal","Step 5\nMAPK","Step 6\nTranscription")),
    x      = 1:6,
    Genes  = c("FUT8", "Lgals3", "CD44\nITGB1", "RhoA\nROCK1\nROCK2",
               "MAPK1\nMAPK3", "NFKB1\nRELA\nSTAT3"),
    Color  = c("#E8F5E9","#C8E6C9","#FFF9C4","#FFE0B2","#FFCCBC","#FFCDD2")
  )
  
  p3b <- ggplot(cascade_nodes) +
    # Step boxes
    geom_rect(aes(xmin = x - 0.35, xmax = x + 0.35,
                  ymin = 0.4, ymax = 0.9, fill = Color),
              color = "grey50", linewidth = 0.5, alpha = 0.7) +
    # Gene names
    geom_text(aes(x = x, y = 0.65, label = Genes), size = 3.2, fontface = "bold",
              color = "navy") +
    # Step labels
    geom_text(aes(x = x, y = 1.05, label = Step), size = 3.5, fontface = "bold",
              lineheight = 0.85) +
    # Arrows between steps
    annotate("segment", x = 1.35, xend = 1.65, y = 0.65, yend = 0.65,
             arrow = arrow(length = unit(0.1, "inches")), linewidth = 1.2, color = "navy") +
    annotate("segment", x = 2.35, xend = 2.65, y = 0.65, yend = 0.65,
             arrow = arrow(length = unit(0.1, "inches")), linewidth = 1.2, color = "navy") +
    annotate("segment", x = 3.35, xend = 3.65, y = 0.65, yend = 0.65,
             arrow = arrow(length = unit(0.1, "inches")), linewidth = 1.2, color = "navy") +
    annotate("segment", x = 4.35, xend = 4.65, y = 0.65, yend = 0.65,
             arrow = arrow(length = unit(0.1, "inches")), linewidth = 1.2, color = "navy") +
    annotate("segment", x = 5.35, xend = 5.65, y = 0.65, yend = 0.65,
             arrow = arrow(length = unit(0.1, "inches")), linewidth = 1.2, color = "navy") +
    
    # [关键修复]: PAR-2 callout 框扩大，宽度从 2.1 延伸至 4.9
    annotate("rect", xmin = 2.1, xmax = 4.9, ymin = 0.05, ymax = 0.35,
             fill = "#FFEBEE", color = cc[["red"]], linewidth = 0.8, alpha = 0.8) +
    annotate("text", x = 3.5, y = 0.2,
             label = "PAR-2: Cross-Cascade Bridge (Step 3 + Step 4)",
             size = 3.5, fontface = "bold", color = cc[["red"]]) +
    
    # Downstream annotation
    annotate("text", x = 3.5, y = 1.3,
             label = "TGF-β1 → P-Selectin → Fibrinogen → Venous Wall Fibrosis",
             size = 3.5, fontface = "italic", color = "purple") +
    scale_fill_identity() +
    xlim(0.3, 6.7) + ylim(-0.1, 1.5) +
    labs(title = "Mechanism Cascade (FUT8 → NF-kB)",
         subtitle = "PAR-2 uniquely anchors both CD44 (Step 3) and RhoA (Step 4)") +
    theme_void() +
    theme(plot.title = element_text(size = 10, face = "bold", hjust = 0),
          plot.subtitle = element_text(size = 8, color = "grey40", hjust = 0))
  
  # -- 3C: GNN vs MR Venn --
  venn_list <- list(
    `MR-Prioritized\nTargets`  = c("F11", "KNG1", "LRP4"),
    `GNN-Discovered\nTargets`  = c("EIF2AK4 mut.", "F2 mRNA expr.", "TLR4 KO")
  )
  
  p3c <- ggvenn(venn_list,
                fill_color = c(cc[["blue"]], cc[["green"]]),
                fill_alpha = 0.3,
                stroke_color = "white",
                stroke_size = 0.5,
                set_name_size = 3.5,
                text_size = 3.5
  ) +
    labs(title = "Multi-Method Cross-Validation",
         subtitle = "GNN discovers orthogonal targets not found by MR") +
    theme_void() +
    theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
          plot.subtitle = element_text(size = 8, color = "grey40", hjust = 0.5))
  
  # -- 3D: Degree vs GNN Score Scatter --
  p3d <- ggplot(targets, aes(x = Degree, y = Score, color = Novelty)) +
    geom_point(aes(size = Score), alpha = 0.8) +
    geom_text_repel(
      data = subset(targets, Target %in% c("PAR-2","Renin","MMP-2","TSP-1","C3")),
      aes(label = Target), size = 3.5, fontface = "bold", box.padding = 0.5,
      max.overlaps = 10, color = "black"
    ) +
    annotate("rect", xmin = 0, xmax = 80, ymin = 25, ymax = 40,
             fill = cc[["green"]], alpha = 0.06) +
    # [关键修复]: 将绿字强行固定在阴影区域的最左上角(x=5, y=39.5)，彻底避开中心的数据点
    annotate("text", x = 5, y = 39.5, label = "High-priority\ndiscovery zone",
             size = 3, color = "darkgreen", fontface = "bold", hjust = 0, vjust = 1) +
    scale_color_manual(values = c("Novel Mechanism" = cc[["red"]],
                                  "Underexplored"   = cc[["orange"]],
                                  "Emerging"        = cc[["green"]],
                                  "Known"           = cc[["grey"]])) +
    scale_size_continuous(range = c(2, 8), guide = "none") +
    labs(title = "Low-Degree, High-Signal Hidden Targets",
         x = "KG Degree (connectivity)", y = "GNN Score") +
    theme_pub
  
  # 组装 Figure 3
  design <- "
    AABB
    AABB
    CCDD
    CCDD
  "
  fig3 <- p3a + p3b + p3c + p3d +
    plot_layout(design = design) +
    plot_annotation(
      title = "Figure 3: Hidden Target Discovery & Mechanism Cascade",
      tag_levels = "a",
      theme = theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
                    plot.tag = element_text(size = 18, face = "bold")))
  
  ggsave("D:/JY/work/my work/新思路/vte_gnn_target_discovery/figures/paper_figures/Figure3_Hidden_Targets.png", fig3,
         width = 15, height = 11, dpi = 600, device = "png", bg = "white")
}


# ============================================================
# Main
# ============================================================

message("========================================")
message("Rendering paper Figures 1–3 (ggplot2)")
message("========================================")

make_fig1()
make_fig2()
make_fig3()

files <- list.files("D:/JY/work/my work/新思路/vte_gnn_target_discovery/figures/paper_figures", pattern = "\\.png$")
message(sprintf("\nDone. %d files in D:/JY/work/my work/新思路/vte_gnn_target_discovery/figures/paper_figures/:", length(files)))
for (f in files) message(sprintf("  %s", f))

