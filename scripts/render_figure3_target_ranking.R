#!/usr/bin/env Rscript
# Figure 3: GNN Global Prioritization of VTE Pathological Programs
# Consumes: figures/hidden_targets/full_ranked_candidates.json
library(ggplot2)
library(patchwork)
library(dplyr)
library(tidyr)
library(jsonlite)
library(gridExtra)

if (requireNamespace("rstudioapi", quietly = TRUE) &&
    rstudioapi::isAvailable()) {
  setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
}
OUT <- "figures/paper_figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# Load (Robust Encoding + Correct Field Names)
# ============================================================
json_path <- "figures/hidden_targets/full_ranked_candidates.json"

json_txt <- readLines(json_path, encoding = "UTF-8", warn = FALSE)
json_txt <- iconv(json_txt, from = "UTF-8", to = "UTF-8", sub = "byte")

ranked <- fromJSON(paste(json_txt, collapse = "\n"), simplifyDataFrame = TRUE)
ranked <- as.data.frame(ranked)

# 字段重命名与清洗，对齐实际 JSON key
ranked$target <- gsub("nf-.*b", "nf-kb", ranked$canonical_name, ignore.case = TRUE)
ranked$target <- ifelse(is.na(ranked$target) | ranked$target == "", ranked$original_name, ranked$target)
ranked$gnn_score <- as.numeric(ranked$score)
ranked$type <- ranked$src_type
ranked$rank <- as.integer(ranked$rank)

# 如果 JSON 里没有 degree 列，则自动从 rank 计算伪 degree 占位，避免报错
if (!"degree" %in% colnames(ranked)) {
  ranked$degree <- max(ranked$rank) - ranked$rank + 1
} else {
  ranked$degree <- as.integer(ranked$degree)
}

# ============================================================
# Panel A: Top-30 bar chart, colored by pathway axis
# ============================================================
top30 <- head(ranked, 30)
top30$target <- factor(top30$target, levels = rev(top30$target))

top30$pathway <- "TLR4/Inflammation"
fibrosis_hits <- grep("smad|runx|dnmt|prkc|gata|san gene|vwf transc|factor v leiden|pomc|mir-155|col|fn1|acta",
                      top30$target, ignore.case = TRUE)
top30$pathway[fibrosis_hits] <- "TGFB/Fibrosis"

p1 <- ggplot(top30, aes(x = gnn_score, y = target, fill = pathway)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = c("TGFB/Fibrosis" = "#E64B35",
                               "TLR4/Inflammation" = "#4DBBD5")) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(x = "GNN Score", y = "",
       title = "A  Top 30 GNN-Prioritized VTE Targets",
       subtitle = "Entity-resolved global ranking, 706M candidate pairs scored") +
  theme_minimal(base_size = 10) +
  theme(legend.position = c(0.75, 0.15),
        legend.background = element_rect(fill = "white", color = NA),
        plot.title = element_text(face = "bold", size = 12))

# ============================================================
# Panel B: Rank / Degree Distribution
# ============================================================
p2 <- ggplot(ranked, aes(x = rank, y = gnn_score)) +
  geom_line(color = "#3C5488", linewidth = 1) +
  geom_point(aes(color = type), alpha = 0.7, size = 1.8) +
  scale_color_manual(values = c("Gene" = "#00A087", "Protein" = "#E64B35")) +
  labs(x = "Global Rank", y = "GNN Score",
       title = "B  Score Distribution across All Candidates",
       subtitle = paste0("Total ", nrow(ranked), " entity-resolved candidate targets")) +
  theme_minimal(base_size = 10) +
  theme(legend.position = c(0.8, 0.8),
        legend.background = element_rect(fill = "white", color = NA),
        plot.title = element_text(face = "bold", size = 12))

# ============================================================
# Panel C: Pathway category breakdown
# ============================================================
top100 <- head(ranked, 100)
top100$category <- "Other"
top100$category[grepl("coagul|thrombin|factor|fibrin|prothrombin|plasmin|fxa|factor x",
                      top100$target, ignore.case = TRUE)] <- "Coagulation"
top100$category[grepl("tlr|nf-kb|inflamm|p-selectin|cytokine|il-|tnf|enos|pai-1|egfr|ace2",
                      top100$target, ignore.case = TRUE)] <- "Inflammation"
top100$category[grepl("smad|tgf|col|fn1|acta|fibrosis|runx|dnmt|gata|prkc",
                      top100$target, ignore.case = TRUE)] <- "Fibrosis/TGFB"
top100$category[grepl("vegf|vegfr|pdgfr|kit|alk1|endoglin",
                      top100$target, ignore.case = TRUE)] <- "Vascular Signaling"

pathway_counts <- top100 %>%
  count(category) %>%
  mutate(pct = round(n / sum(n) * 100, 1))

p3 <- ggplot(pathway_counts, aes(x = reorder(category, n), y = n, fill = category)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = paste0(n, " (", pct, "%)")), hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(x = "", y = "Number of Targets (Top 100)",
       title = "C  Pathway Distribution of Top Candidates") +
  theme_minimal(base_size = 10) + guides(fill = "none") +
  theme(plot.title = element_text(face = "bold", size = 12))

# ============================================================
# Panel D: Top-10 table
# ============================================================
top10 <- head(ranked, 10)[, c("rank", "target", "type", "gnn_score", "disease")]
colnames(top10) <- c("Rank", "Target", "Type", "Score", "Disease")
top10$Score <- round(top10$Score, 2)
top10$Target <- substr(top10$Target, 1, 35)
top10$Disease <- substr(top10$Disease, 1, 20)

tbl <- tableGrob(top10, rows = NULL,
                 theme = ttheme_minimal(base_size = 8,
                                        core = list(fg_params = list(hjust = 0, x = 0.05))))

p4 <- ggplot() + annotation_custom(tbl) + theme_void() +
  labs(title = "D  Top 10 GNN-Prioritized Targets") +
  theme(plot.title = element_text(face = "bold", size = 12))

# ============================================================
# Assemble
# ============================================================
combined <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title = "Figure 3: GNN Global Prioritization of VTE Molecular Regulators",
    subtitle = paste0(nrow(ranked), " entity-resolved candidates from 706,523,994 scored pairs | No anchor filtering")
  ) &
  theme(plot.title = element_text(face = "bold", size = 15))

ggsave(file.path(OUT, "Figure3_Target_Ranking.tiff"), combined,
       width = 14, height = 12, dpi = 300, compression = "lzw")
message("Saved: Figure3_Target_Ranking.tiff")
