#!/usr/bin/env Rscript
# Supplementary Figure 1: Cross-species transcriptomic validation
# GSEA of mouse PAR-2+ fibroblast signatures in human VTE blood transcriptomes

library(ggplot2)
library(dplyr)

# ── 1. Data Preparation ───────────────────────────────────────────
gsea <- data.frame(
  Signature = rep(c("PAR-2+ Activated\nFibroblast Up",
                    "PAR-2+ Quiescent\nFibroblast Up"), each = 2),
  Dataset = rep(c("GSE48000\n(107 VTE, 25 controls)",
                  "GSE19151\n(70 VTE, 63 controls)"), times = 2),
  NES = c(1.65, -1.50, 1.57, 0.95),
  FDR = c(0.016, 0.037, 0.017, 0.553),
  stringsAsFactors = FALSE
)

# 设定因子水平，控制排列顺序
gsea$Signature <- factor(gsea$Signature,
                         levels = c("PAR-2+ Quiescent\nFibroblast Up",
                                    "PAR-2+ Activated\nFibroblast Up"))
gsea$Dataset <- factor(gsea$Dataset,
                       levels = c("GSE48000\n(107 VTE, 25 controls)",
                                  "GSE19151\n(70 VTE, 63 controls)"))

# ── 2. Nature-Style Colors ────────────────────────────────────────
RED   <- "#D55E00"
BLUE  <- "#0072B2"

# ── 3. Plotting ───────────────────────────────────────────────────
p <- ggplot(gsea, aes(x = NES, y = Signature, fill = Dataset)) +
  # 添加零点参考线
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.6) +
  
  # 绘制散点，使用 position_dodge 将两个数据集错开
  # 气泡大小与 -log10(FDR) 挂钩，显著性越强，点越大
  geom_point(aes(size = -log10(FDR)), 
             shape = 21, color = "black", stroke = 0.6,
             position = position_dodge(width = 0.4)) +
  
  # 颜色映射
  scale_fill_manual(values = c("GSE48000\n(107 VTE, 25 controls)" = RED,
                               "GSE19151\n(70 VTE, 63 controls)" = BLUE)) +
  
  # 设置 X 轴范围，留出充足边缘
  scale_x_continuous(limits = c(-2.0, 2.0), breaks = seq(-2, 2, 1)) +
  
  # 精调气泡大小图例
  scale_size_continuous(range = c(2, 7), 
                        breaks = c(-log10(0.5), -log10(0.05), -log10(0.01)),
                        labels = c("NS (0.50)", "0.05", "0.01"),
                        name = "FDR") +
  
  # 标题与标签：巧妙利用 x 轴标题指代方向，避免错位
  labs(
    x = "← Favors Control                    Normalized Enrichment Score (NES)                    Favors VTE →", 
    y = NULL, 
    fill = "Cohort",
    title = "Supplementary Figure 1: Cross-species transcriptomic validation",
    subtitle = "Significant enrichment of PAR-2+ fibroblast signatures in human VTE (GSE48000)"
  ) +
  
  # Nature 通用主题精调
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12, margin = margin(b = 5)),
    plot.subtitle = element_text(size = 9, color = "grey30", margin = margin(b = 15)),
    axis.text.y = element_text(face = "bold", color = "black", lineheight = 1.2),
    axis.title.x = element_text(face = "bold", margin = margin(t = 10)),
    legend.position = "right",
    legend.background = element_rect(fill = alpha("white", 0.8), color = "grey80", linewidth = 0.3),
    legend.margin = margin(6, 6, 6, 6),
    panel.grid.major.x = element_line(color = "grey90", linewidth = 0.3)
  )

# ── 4. Save Output ────────────────────────────────────────────────
ggsave("Supplementary_Figure_1.png", plot = p, width = 8.5, height = 4.5, dpi = 300, bg = "white")
ggsave("Supplementary_Figure_1.pdf", plot = p, width = 8.5, height = 4.5, bg = "white")

cat("Supplementary Figure 1 rendered successfully.\n")

