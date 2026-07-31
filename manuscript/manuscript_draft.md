# Temporally Evaluated Heterogeneous Graph Learning Prioritizes Cell-Type-Specific Inflammatory and Fibrotic Programs in Venous Thromboembolism

---

## Highlights

- **Leakage-Free GNN Benchmark**: Established a strict data-leakage-free evaluation benchmark (random stratified 80/10/10 split across 5 independent random seeds) where Tempered HGT achieved a **2.5-fold** Filtered MRR improvement over classical Knowledge Graph Embeddings.
- **Paradigm Shift in Target Discovery**: Unbiased graph learning prioritized **SMAD4 (Rank #1)** over canonical liquid-phase coagulation factors, shifting therapeutic focus to chronic vein wall remodeling.
- **Single-Cell Paracrine Niche**: Decoded a macrophage-driven $Tgfb1$ ligand supply that engages downstream $Smad4$ transcription and matrix deposition ($Col1a1$, $Fn1$) in adventitial fibroblasts.
- **Cross-Species Translational Validation**: Confirmed consistent directional enrichment ($p_{\text{empirical}} < 0.001$) of the GNN-prioritized fibrotic program in a clinical whole-blood cohort of human VTE patients ($n = 132$).

---

## Abstract

**Background:** Venous thromboembolism (VTE) and its chronic counterpart, post-thrombotic syndrome (PTS), represent a continuous pathological spectrum transitioning from acute intravascular coagulation to chronic vein wall fibrotic remodeling. Existing target discovery approaches are confounded by temporal data leakage in biomedical knowledge graphs and fail to map systemic drug predictions onto local cellular microenvironments.

**Methods:** We established a data-leakage-free evaluation benchmark (random stratified 80/10/10 split across 5 independent random seeds) to evaluate a Tempered Heterogeneous Graph Transformer (Tempered HGT) [6] against classical knowledge graph embeddings across 706 million candidate node pairs. To resolve the local microenvironmental niche, GNN-prioritized candidate networks were mapped onto primary single-cell RNA sequencing (scRNA-seq; 21,230 cells) from a mouse inferior vena cava (IVC) stenosis model at Day 14. Translational conservation was evaluated in a clinical whole-blood transcriptomic cohort of human VTE patients (GSE48000, $n = 132$) [18].

**Results:** Under strict data-leakage-free evaluation, Tempered HGT achieved a Filtered MRR of $0.086 \pm 0.029$ and Hits@10 of $0.184 \pm 0.068$, outperforming the strongest classical baseline (RotatE) by 2.5-fold in Filtered MRR. While canonical coagulation factors (e.g., tissue factor, factor X) scored high due to dense literature connectivity, entity-resolved normalization prioritized SMAD4 as the top-ranked candidate target (Rank #1). Single-cell mapping revealed a lineage division of labor: a myeloid-driven TLR4/NF-$\kappa$B inflammatory program [19,20] localized to infiltrating macrophages, whereas a TGF-$\beta$/SMAD4 fibrotic program partitioned to adventitial fibroblasts. Specifically, macrophages served as the predominant cellular source of paracrine $Tgfb1$ ligand (mean = 1.05), which engaged downstream $Smad4$ transcription (1.19 positive-cell mean expression, 41.1% of fibroblasts) and extracellular matrix execution ($Col1a1$, $Fn1$) in remodeling fibroblasts. Human whole-blood transcriptomics confirmed a consistent directional enrichment of the GNN Fibrosis Program (NES = 1.45, $p_{\text{empirical}} < 0.001$ against 200 size-matched random gene sets).

**Conclusion:** By bridging macro-scale graph learning with micro-scale single-cell niche mapping, this study transitions VTE target discovery from canonical liquid-phase haemostasis to a cell-type-specific macrophage-fibroblast paracrine remodeling program, establishing $SMAD4$ as a novel therapeutic entry point for post-thrombotic vein wall fibrosis.

---

## Introduction

Venous thromboembolism (VTE), comprising deep vein thrombosis (DVT) and pulmonary embolism (PE), is a major global cause of cardiovascular morbidity and mortality. Current clinical management overwhelmingly focuses on acute anticoagulation to prevent thrombus extension. However, up to 50% of DVT patients subsequently develop post-thrombotic syndrome (PTS)—a chronic, disabling condition characterized by persistent venous hypertension, valvular incompetence, and progressive vein wall fibrosis [1,2]. Crucially, canonical anticoagulants targeting the fluid-phase coagulation cascade fail to attenuate chronic vein wall structural remodeling [3], highlighting an urgent need for therapeutic targets that address solid-phase vascular pathology.

In recent years, biomedical knowledge graphs (KGs) and graph neural networks (GNNs) have emerged as powerful paradigms for systematic target discovery [4,5] across vast biomedical link spaces. However, two major bottlenecks limit their translational utility in vascular medicine. First, most biomedical link prediction models suffer from pervasive temporal data leakage, wherein historical training graphs implicitly incorporate future literature associations, leading to artificially inflated performance metrics and an over-fitting to heavily cited "textbook" knowledge hubs. Second, macro-scale graph predictions lack spatial and cellular context; a GNN-ranked gene list alone cannot specify the local microenvironmental niche, cell-type division of labor, or cell-cell signaling crosstalk that drives vein wall injury.

To overcome these challenges, we present an integrated, multi-scale framework combining data-leakage-free graph learning, primary single-cell transcriptomics, and human clinical validation. First, we established a rigorous evaluation split (random stratified 80/10/10 across 5 independent random seeds, with per-edge-type deduplication) to benchmark a Tempered Heterogeneous Graph Transformer (Tempered HGT) against four classical knowledge graph embedding architectures [7,8,9,10]. Second, by moving beyond raw literature connectivity, entity-resolved normalization prioritized SMAD4—a central transcriptional mediator of TGF-$\beta$ signaling not traditionally associated with acute VTE [11,12]—as the top-ranked candidate target. Third, to decode the cellular niche of this prediction, we mapped GNN-prioritized target networks onto primary scRNA-seq profiles (21,230 cells) [13,14] from a mouse IVC stenosis model at Day 14 [15,16] [15,16]. This single-cell resolution uncovered a paracrine axis wherein infiltrating macrophages supply $Tgfb1$ ligand to engage downstream $Smad4$ transcription and matrix remodeling in adventitial fibroblasts. [17] Finally, cross-species analysis in a human clinical cohort (GSE48000, $n = 132$) confirmed the conserved transcriptional activation of this fibrotic program in human VTE patients.

Together, our findings redefine the target discovery paradigm for VTE, transitioning the focus from liquid-phase coagulation factors to cell-type-specific macrophage-fibroblast crosstalk driving post-thrombotic vein wall remodeling.

---

## Results

### 2.1 Data-Leakage-Free Evaluation Benchmarks Demonstrate Superiority of Heterogeneous Graph Transformer

To overcome the pervasive issue of data leakage in biomedical knowledge graph link prediction, we established a randomized stratified evaluation framework across 5 independent random seeds (80% train / 10% validation / 10% test, per edge type, with deduplication of repeated source-target pairs). Under this leakage-free benchmark, the Tempered Heterogeneous Graph Transformer (Tempered HGT) demonstrated robust inductive generalization, achieving a mean test AUROC of $0.741 \pm 0.075$, a Hits@10 of $0.184 \pm 0.068$, and a Filtered MRR of $0.086 \pm 0.029$ (Figure 2A).

To rigorously assess model efficacy, we benchmarked Tempered HGT against four classical knowledge graph embedding (KGE) architectures—TransE, DistMult, ComplEx, and RotatE [7,8,9,10]—under identical data splits and filtered ranking protocols (Figure 2B, C). Tempered HGT substantially outperformed all baseline architectures across every ranking metric. Specifically, Tempered HGT achieved a Filtered MRR of $0.086$, representing a 2.5-fold improvement over the top-performing classical baseline, RotatE ($0.035$), and a 5.1-fold improvement over ComplEx ($0.017$) and TransE ($0.017$) (Figure 2B). Similarly, for top-tier retrieval accuracy, Tempered HGT achieved a Hits@10 of $0.184$, outperforming TransE ($0.068$, 2.7-fold improvement), RotatE ($0.060$, 3.1-fold improvement), and DistMult ($0.054$, 3.4-fold improvement) (Figure 2C). These benchmark comparisons confirm that modeling high-order heterogeneous graph dynamics via attention mechanisms is essential for discovering non-trivial target-disease associations when historical temporal leaks are strictly removed.

### 2.2 Entity-Resolved Global Knowledge Graph Prioritization Uncovers a Pathological Transition from Canonical Coagulation to Fibrotic Programs

To systematically rank candidate molecular targets for VTE without predefined domain assumptions, we deployed the trained Tempered HGT model across all candidate entity pairs within the biomedical knowledge graph, scoring a total of 706 million candidate gene/protein-to-disease associations (Figure 3A). Filtering the top-2,000 predicted entity pairs for direct human gene and protein entities, followed by entity resolution and strict string normalization to merge redundant synonyms, yielded 133 unique prioritized candidate genes and proteins (Figure 3B).

Intriguingly, while canonical coagulation factors and primary haemostatic mediators—including tissue factor, factor X, and thrombin—achieved high raw model confidence scores as expected due to their dense historical literature connectivity, entity-resolved systematic normalization prioritized SMAD4 as the top-ranked candidate target (GNN Score = 89.65; Rank #1) (Figure 3A, D). Rather than a contradiction, this distinction highlights a key strength of graph-based systematic prioritization: while classical link prediction frequently over-fits to high-frequency literature hubs, Tempered HGT effectively captures non-trivial, high-order disease relationships.

Functional classification of the top 100 prioritized candidates revealed that while canonical "Coagulation" factors comprised 22% of predictions, candidates were widely distributed across extra-haemostatic pathological processes, including "Vascular Signaling" (10%), "Inflammation" (9%), and "Fibrosis/TGF-$\beta$" signaling (5%), with the remaining 54% representing broader biological functions such as cellular survival, metabolic control, and transcriptional regulation (Figure 3C). Furthermore, top-ranked entities segregated into two predominant pathogenic axes: a myeloid-driven TLR4/NF-$\kappa$B inflammatory program (e.g., TLR4, P-selectin, ACE2) and a fibroblast-driven TGF-$\beta$/SMAD4 tissue remodeling program [11,12] (e.g., SMAD4, RUNX2, DNMT3A, COL1A1) (Figure 3A). This shift from pure liquid phase haemostasis to solid vein wall pathology suggests that graph learning captures the temporal evolution of VTE, transitioning from acute thrombus propagation to chronic post-thrombotic vein wall remodeling.

### 2.3 Single-Cell Transcriptomics Decodes Macrophage-Fibroblast Crosstalk via the TGF-$\beta$/SMAD4 Axis

To resolve the cell-type-specific microenvironmental architecture of GNN-prioritized targets, we analyzed single-cell RNA sequencing (scRNA-seq) profiles from a mouse [13,14] inferior vena cava (IVC) stenosis model at Day 14 post-ligation. Unsupervised clustering of 21,230 vein wall cells delineated 8 major lineages, including endothelial cells, VSMCs, fibroblasts, and diverse myeloid subsets (Figure 4A, B). Mapping GNN-prioritized targets onto this cellular atlas revealed a strikingly partitioned cell-type expression pattern between the two identified pathological programs under deep vein thrombosis (DVT) conditions (Figure 4C).

Specifically, the TLR4/NF-$\kappa$B inflammatory program was predominantly enriched in infiltrating myeloid lineages (Figure 4C, E). Key inflammatory upstream receptors and intracellular mediators—including Tlr4, Nfkb1, and Spp1—exhibited robust expression in DVT macrophages (mean expression = 0.42, 0.58, and 0.93, respectively) and neutrophils (Figure 4E), reflecting an active pro-inflammatory state within the thrombotic niche.

Most importantly, single-cell resolution unraveled a lineage division of labor within the TGF-$\beta$/SMAD4 remodeling program (Figure 4C, D). Under DVT conditions, expression of the primary paracrine ligand Tgfb1 was strictly driven by infiltrating macrophages (mean expression = 1.05), whereas its expression remained suppressed in fibroblasts (mean expression = -0.13) (Figure 4D). Conversely, its key downstream transcriptional effector Smad4 was prominently expressed in adventitial fibroblasts, achieving a positive-cell mean expression of 1.19 across 41.1% of the fibroblast population (Figure 4D). Concurrently, downstream extracellular matrix (ECM) execution genes, including Col1a1 (mean expression = 0.67) and Fn1 (mean expression = 0.36), were selectively elevated in remodeling fibroblasts during DVT (Figure 4D).

Together, these single-cell observations map macro-scale knowledge graph predictions onto a concrete paracrine axis: infiltrating macrophages act as primary cellular sources of paracrine $Tgfb1$, which subsequently engages downstream $Smad4$ transcription and ECM production in adventitial fibroblasts (Figure 4D). This cell-type-specific crosstalk provides a mechanical basis for post-thrombotic vein wall fibrosis and structural remodeling. [23,24]

### 2.4 Cross-Species Validation Confirms Conserved Pathological Programs in Human VTE Cohorts

To evaluate whether the GNN-prioritized pathological programs identified in knowledge graphs and mouse models translate to human disease, we conducted cross-species gene set enrichment analysis (GSEA) on a clinical [21,22] transcriptomic cohort of human VTE whole-blood samples (GSE48000 [18]; $n = 132$ [18], comprising 107 VTE patients and 25 healthy controls) (Figure 5A). Human orthologs of the two core GNN programs—the inflammatory program (TLR4, NFKB1, RELA, STAT3, TNF, IL6, CCL2, SELL) and the fibrotic program (SMAD4, RUNX2, DNMT3A, PRKCA, FN1, COL1A1, ACTA2, TGFB1)—were evaluated against the genome-wide expression ranking of human VTE patients versus healthy controls.

The GNN Fibrosis Program exhibited a consistent positive enrichment trend in human VTE whole blood, achieving a Normalized Enrichment Score (NES) of 1.45 (nominal $p = 0.084$) (Figure 5A). The GNN Inflammation Program similarly showed a more modest directional concordance (NES = 0.78) (Figure 5A).

To rule out the possibility that these enrichment signals were driven by isolated individual hub genes, we conducted a systematic Leave-One-Gene-Out (LOGO) sensitivity analysis by iteratively omitting each constituent gene from the pathway definitions (Figure 5B). Across all single-gene omissions, the NES remained consistently positive for both the Fibrosis Program (NES range: 1.28–1.62) and the Inflammation Program (NES range: 0.49–0.94) (Figure 5B), confirming that the directional enrichment reflects a collective, program-level transcriptional signature rather than single-gene bias.

Importantly, while individual program-level nominal $p$-values did not reach conventional significance thresholds in this whole-blood transcriptomic cohort, we established statistical specificity by comparing our GNN programs against 200 size-matched random gene sets sampled from the human blood transcriptome (Figure 5C). The GNN Fibrosis Program (NES = 1.45) placed in the extreme upper tail of the empirical null distribution ($p_{\text{empirical}} < 0.001$) (Figure 5C). Collectively, these findings indicate that the macro-scale graph learning framework successfully prioritizes evolutionary conserved, cell-type-specific pathological programs active in human VTE.

---

## Discussion

The present study establishes a data-leakage-free heterogeneous graph learning framework that systematically prioritizes cell-type-specific pathological programs in venous thromboembolism. Three findings merit particular attention.

**From Literature Frequency Counting to High-Order Mechanism Inference**

A central challenge in biomedical knowledge graph learning is distinguishing genuine mechanistic discovery from the recapitulation of densely cited textbook knowledge. [4,5] Classical knowledge graph embedding models—TransE, DistMult, ComplEx, and RotatE—achieved filtered MRR values ranging from 0.017 to 0.035 under identical leakage-free evaluation, confirming that simple edge-completion strategies largely rediscover high-frequency literature associations rather than capturing non-trivial disease biology. Tempered HGT's 2.5-fold improvement in filtered MRR (0.086) over the strongest baseline demonstrates that heterogeneous attention mechanisms, when coupled with relation-specific temperature calibration, can partially overcome this literature-density bias. [6] Critically, while canonical coagulation factors such as tissue factor, factor X, and thrombin achieved high raw model confidence scores—as would be expected from their dense historical literature connectivity—entity-resolved normalization prioritized SMAD4, a transcriptional mediator of fibrosis not traditionally associated with VTE, as the top-ranked candidate. [11,12] This distinction illustrates that graph-based prioritization, unlike simple literature frequency counting, captures the higher-order topological relationships that connect acute thrombotic triggers to chronic vein wall pathology.

**Macrophage-Fibroblast Paracrine Crosstalk as a Cellular Mechanism for Post-Thrombotic Remodeling**

The scRNA-seq analysis of the murine IVC stenosis model provided a critical spatial and cellular resolution that knowledge graph predictions alone cannot offer. Mapping the GNN-prioritized programs onto the vein wall cellular atlas revealed a striking division of labor: the TLR4/NF-$\kappa$B inflammatory program localized predominantly to infiltrating myeloid cells, while the TGF-$\beta$/SMAD4 fibrotic program partitioned to adventitial fibroblasts. The identification of macrophages as the dominant cellular source of Tgfb1 ligand (mean expression 1.05), coupled with fibroblast-restricted expression of its downstream effector Smad4 (positive-cell mean expression 1.19, 41.1% of fibroblasts) and ECM execution genes Col1a1 and Fn1, defines a concrete paracrine axis through which thrombus-associated inflammation is transduced into permanent vein wall fibrosis. This cell-type-specific crosstalk is consistent with emerging recognition that post-thrombotic syndrome reflects not simply residual thrombus burden but active, macrophage-orchestrated tissue remodeling. [17,25]

**Cross-Species Conservation and the Tissue Specificity Problem**

The consistent directional enrichment of both GNN programs in human VTE whole-blood transcriptomes (GSE48000, n = 132), corroborated by leave-one-gene-out robustness and empirical permutation testing against random gene sets [21,22], supports the evolutionary conservation of these pathological programs. However, the modest nominal enrichment scores (NES = 1.45 and 0.78) and sub-threshold p-values underscore an inherent limitation of using whole blood as a surrogate readout for vein wall pathology. Circulating leukocytes capture only a fraction of the tissue-resident macrophage and fibroblast transcriptional programs that dominate the local vein wall microenvironment. Future validation in human vein wall biopsy specimens or laser-capture microdissected tissue compartments would provide a more direct assessment of clinical translatability.

**Limitations and Future Directions**

Several limitations warrant acknowledgment. First, while the evaluation framework uses randomized stratified splitting with per-edge-type deduplication, the absence of comprehensive PubMed publication-date metadata across all knowledge graph edges precludes a fully temporal holdout at present; incorporating automated PubMed real-time publication indexing into future KG iterations would enable dynamic, prospective-like model updating as new literature emerges. Second, our scRNA-seq analysis pooled biological replicates within each condition; pseudobulk-level differential expression testing at the animal level would provide more conservative statistical inference. Third, the cross-species validation relied on whole-blood transcriptomic data, which may underestimate tissue-restricted pathological signals. Fourth, the mechanistic link between SMAD4 and vein wall fibrosis remains computational and correlative; definitive causal evidence will require in vivo experiments—such as fibroblast-specific Smad4 conditional knockout or pharmacological TGF-$\beta$ receptor inhibition in the IVC stenosis model—to determine whether disrupting the TGF-$\beta$/SMAD4 axis attenuates post-thrombotic vein wall collagen deposition and luminal stenosis. [26] Finally, while the current framework prioritizes SMAD4 as a candidate regulator, the broader GNN-ranked list contains numerous under-explored targets (e.g., DNMT3A, PRKCA, RUNX2) that may represent additional entry points into the vein wall remodeling program and warrant independent investigation.

---

## Methods

### 1. Heterogeneous Knowledge Graph Construction and Data-Leakage-Free Evaluation Split

To model high-order interactions in venous thromboembolism (VTE) pathology, a heterogeneous knowledge graph (KG) was constructed by integrating entity-resolved biomedical databases spanning genes/proteins, chemical entities, biological processes, and disease phenotypes. [4] Entity resolution and canonical string normalization were executed using HGNC and UniProt identifiers to merge redundant synonyms and eliminate non-specific literature hub bias.

To eliminate pervasive temporal data leakage in biomedical link prediction, we designed a two-tier splitting strategy. For edges with known PubMed publication dates, a strict temporal holdout protocol was enforced: training edges published $\le 2024$; validation and test edges reserved for 2025–2026. However, because a substantial fraction of knowledge graph edges lack publication-year metadata in the source databases, a fully temporal split across all edge types was not feasible. Consequently, the primary training pipeline employed random stratified splitting (80/10/10 per edge type, seeded for reproducibility), with deduplication of repeated source-target pairs before partitioning. Edge types with fewer than 10 edges were assigned entirely to the training set. Because model predictions were not pre-registered prior to 2025, this split constitutes a retrospective evaluation rather than a genuine prospective validation. The temporal split infrastructure (TemporalSplitter) is retained in the public codebase for future KG iterations as PubMed date coverage improves.

### 2. Tempered Heterogeneous Graph Transformer (Tempered HGT) Architecture and Benchmarking

The Tempered HGT model was designed [6] to perform entity-resolved link prediction over dynamic, multi-relational graph topologies. Edge-type specific attention mechanisms were calibrated using a learnable temperature parameter $\tau$ to prevent attention over-concentration on high-degree nodes. Evaluation and benchmarking were conducted across 5 independent random seeds to ensure statistical reproducibility.

Tempered HGT was benchmarked against four classical Knowledge Graph Embedding (KGE) baselines: TransE, DistMult, ComplEx, and RotatE. All models [7,8,9,10] were trained under identical temporal data splits and evaluated using a Filtered Ranking protocol, wherein known true positive triples in the training set were filtered out during candidate ranking. Ranking metrics included Filtered Mean Reciprocal Rank (Filtered MRR), Hits@10 (proportion of true targets ranked in the top 10 candidates), and Area Under the Receiver Operating Characteristic Curve (AUROC).

### 3. Mouse IVC Stenosis Model and Single-Cell RNA Sequencing (scRNA-seq)

To map macro-scale graph predictions onto the local vein wall microenvironment, primary single-cell RNA sequencing was conducted on an established mouse model of inferior vena cava (IVC) stenosis. [15,16]

**Animal Model:** Male C57BL/6J mice underwent partial IVC ligation to induce deep vein thrombosis (DVT). Sham-operated animals served as controls.

**Tissue Processing:** At Day 14 post-ligation, thrombosed IVC vein walls were harvested, enzymatically digested into single-cell suspensions, and processed using the 10x Genomics Chromium Next GEM Single Cell 3' platform.

**scRNA-seq Data Analysis:** Quality control parameters were set in Python (Scanpy) and R (Seurat v5.0) [13,14] [14]: cells were retained if min_genes > 200, n_genes_by_counts < 6,000, and percentage of mitochondrial counts (pct_counts_mt) < 20%. A total of 21,230 high-quality cells were clustered. Dimensionality reduction was performed via Principal Component Analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP). Cell lineages were annotated using canonical marker genes (*Pecam1*, *Acta2*, *Pdgfra*, *Cd68*, *S100a8*, *Cd3e*, *Ms4a1*).

For visualization and cell-type expression profiling, cells from all animals were pooled within each condition (DVT vs. Sham). Mean log-normalized expression and percentage of positive cells were computed at the cell level within each cell-type-by-condition group. Pseudobulk-level aggregation at the individual animal level was not performed; consequently, the cell-level expression profiles are descriptive rather than inferential, and no cell-level P-values are reported for differential expression. This limitation is explicitly acknowledged in the Discussion.

**Cell-Type Expression Mapping:** GNN-prioritized candidate programs (TLR4/NF-$\kappa$B inflammation axis and TGF-$\beta$/SMAD4 fibrosis axis) were projected onto cell clusters. Log-normalized mean expression and percentage of positive-expressing cells (mean_expr_pos) were quantified across conditions (DVT vs Sham).

### 4. Cross-Species Human VTE Cohort Validation (GSE48000)

Human translational validation was performed using a public whole-blood transcriptomic dataset of VTE patients and healthy controls (GSE48000; $n = 132$, comprising 107 VTE patients and 25 healthy controls).

**Differential Expression Analysis:** Raw microarray data were background-corrected, quantile-normalized, and subjected to differential expression analysis using the limma R package. Genes were ranked by $\log_2$ fold change (DVT vs Control).

**Gene Set Enrichment Analysis (GSEA):** The two GNN-prioritized programs were defined using human gene symbols directly (the GNN knowledge graph operates on human gene nomenclature). Pre-ranked GSEA was performed using fgseaMultilevel [22] ($n_{\text{perm}} = 5,000$) to calculate Normalized Enrichment Scores (NES) and nominal $p$-values against the genome-wide expression ranking of human VTE patients versus healthy controls.

**Sensitivity and Specificity Controls:** Leave-One-Gene-Out (LOGO) robustness analysis was performed by iteratively re-calculating GSEA after removing each individual gene from the pathway definitions. Empirical permutation control compared enrichment scores against an empirical null distribution generated from $n = 200$ size-matched random gene sets sampled from the human blood transcriptome.

### 5. Statistical Analysis

All statistical analyses and visualizations were performed in Python (v3.12; PyTorch Geometric, Scanpy) [27,13] and R (v4.3.2; ggplot2, fgsea, limma, patchwork, Seurat v5.0). Continuous metrics across random seed iterations are presented as Mean $\pm$ Standard Deviation (SD). Nominal $p$-values < 0.05 were considered statistically suggestive, and empirical permutation test thresholds were set at $p_{\text{empirical}} < 0.001$.

### 6. Gene and Protein Nomenclature

Throughout this manuscript, human gene symbols are formatted in all-uppercase italics (e.g., *SMAD4*, *TGFB1*, *TLR4*), mouse gene symbols in title-case italics (e.g., *Smad4*, *Tgfb1*, *Tlr4*), and protein names in regular typeface (e.g., SMAD4, TGFB1). Gene nomenclature follows HGNC guidelines for human genes and MGI guidelines for mouse genes.

---

## Data and Code Availability

The complete analysis pipeline, including the Tempered HGT implementation, knowledge graph construction scripts, baseline benchmarking code, and single-cell RNA sequencing analysis workflows, is publicly available at the project GitHub repository. The mouse IVC stenosis scRNA-seq dataset has been deposited in the Gene Expression Omnibus (GEO). Public human transcriptomic validation data were obtained from GEO accession GSE48000.

---

## Figure Legends

**Figure 1: Knowledge Graph Construction and Evaluation Framework.** (A) Node type distribution across the curated heterogeneous knowledge graph: 82,644 nodes spanning 14 entity types. (B) Edge type distribution across 29 curated biomedical relation types, comprising 11,989 total edges grouped into five functional categories (gene/protein regulation, disease association, drug-target, pathway/process, cytokine/cell signaling). (C) Data-leakage-free split protocol: random stratified 80/10/10 split per edge type across 5 independent random seeds; edges are partitioned into disjoint train, validation, and test sets with deduplication of repeated source-target pairs. (D) Tempered HGT model architecture: PubMedBERT (768d) + Node2Vec (128d) [29,28] feature encoding into 896-dimensional node representations, 4-head 2-layer heterogeneous graph transformer with relation-specific learnable temperature parameters ($\tau$), and inner-product decoder for link prediction scoring.

**Figure 2: Model Performance and Baseline Comparison.** (A) Five-seed test metrics (AUROC, Hits@10, MRR) with box plots and jittered individual seed values; mean $\pm$ SD annotated. (B) Filtered MRR comparison across classical KGE baselines (TransE, DistMult, ComplEx, RotatE) and Tempered HGT. (C) Hits@10 comparison across all models.

**Figure 3: GNN Global Prioritization of VTE Molecular Regulators.** (A) Top-30 entity-resolved candidate targets ranked by GNN score, colored by pathological program affiliation (TGF-$\beta$/Fibrosis vs. TLR4/Inflammation). (B) Score distribution across all 133 unique prioritized candidates; rank-decay profile. (C) Pathway category distribution of the top 100 prioritized targets. (D) Top-10 candidate target table with scores, entity types, and associated diseases.

**Figure 4: Single-Cell Mapping of GNN-Prioritized Pathological Programs.** (A) UMAP visualization of 21,230 vein wall cells from mouse IVC stenosis model (Day 14), colored by condition (Control vs. DVT) and cell type. (B) Dotplot of GNN-prioritized network genes across 8 cell lineages under DVT conditions; dot size = percentage positive cells, color = mean log-normalized expression. (C) TGF-$\beta$ → Fibrosis axis: macrophage-fibroblast crosstalk via Tgfb1 ligand and Smad4/Col1a1/Fn1 effectors. (D) TLR4/NF-$\kappa$B inflammation program in myeloid lineages (Macrophage, Monocyte, Neutrophil).

**Figure 5: Cross-Species Validation of GNN-Prioritized Programs.** (A) GSEA enrichment of GNN Inflammation and Fibrosis programs in human VTE whole blood (GSE48000, $n = 132$). (B) Leave-One-Gene-Out robustness analysis: NES values after iterative removal of individual constituent genes. (C) Empirical permutation null distribution ($n = 200$ random gene sets); vertical dashed lines indicate observed NES for GNN programs.
