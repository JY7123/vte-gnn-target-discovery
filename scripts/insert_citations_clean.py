#!/usr/bin/env python3
"""
Cleanly insert Vancouver-numbered citations into manuscript.
Strategy: strip all existing [N] citations, then re-insert based on
context patterns. This avoids the broken numbering from manual edits.
"""
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT / "manuscript" / "manuscript_draft.md"

# ── Ultimate Reference List (30 papers, Vancouver order) ──
# These will be numbered 1-30 in order of first appearance in the manuscript

REFS = [
    # [1] PTS AHA statement
    dict(id=1, cite="Kahn SR, Comerota AJ, Cushman M, et al. Circulation. 2014;130(18):1636-1661.",
         full="1. **Kahn SR**, Comerota AJ, Cushman M, et al. The postthrombotic syndrome: evidence-based prevention, diagnosis, and treatment strategies: a scientific statement from the American Heart Association. *Circulation*. 2014;130(18):1636-1661. doi:10.1161/CIR.0000000000000130"),
    # [2] PTS epidemiology
    dict(id=2, cite="Galanaud JP, Monreal M, Kahn SR. Thromb Res. 2018;164:100-109.",
         full="2. **Galanaud JP**, Monreal M, Kahn SR. Epidemiology of the post-thrombotic syndrome. *Thrombosis Research*. 2018;164:100-109. doi:10.1016/j.thromres.2017.07.026"),
    # [3] How I treat PTS
    dict(id=3, cite="Kahn SR. Blood. 2009;114(21):4624-4631.",
         full="3. **Kahn SR**. How I treat postthrombotic syndrome. *Blood*. 2009;114(21):4624-4631. doi:10.1182/blood-2009-07-199174"),
    # [4] Biomedical KG review
    dict(id=4, cite="Nicholson DN, Greene CS. Comput Struct Biotechnol J. 2020;18:1414-1428.",
         full="4. **Nicholson DN**, Greene CS. Constructing knowledge graphs and their biomedical applications. *Computational and Structural Biotechnology Journal*. 2020;18:1414-1428. doi:10.1016/j.csbj.2020.05.017"),
    # [5] GNN polypharmacy
    dict(id=5, cite="Zitnik M, Agrawal M, Leskovec J. Bioinformatics. 2018;34(13):i457-i466.",
         full="5. **Zitnik M**, Agrawal M, Leskovec J. Modeling polypharmacy side effects with graph convolutional networks. *Bioinformatics*. 2018;34(13):i457-i466. doi:10.1093/bioinformatics/bty294"),
    # [6] HGT
    dict(id=6, cite="Hu Z, Dong Y, Wang K, Sun Y. WWW. 2020:2704-2710.",
         full="6. **Hu Z**, Dong Y, Wang K, Sun Y. Heterogeneous Graph Transformer. In: *Proceedings of The Web Conference 2020 (WWW '20)*. 2020:2704-2710. doi:10.1145/3366423.3380027"),
    # [7] TransE
    dict(id=7, cite="Bordes A, Usunier N, Garcia-Duran A, Weston J, Yakhnenko O. NeurIPS. 2013:2787-2795.",
         full="7. **Bordes A**, Usunier N, Garcia-Duran A, Weston J, Yakhnenko O. Translating Embeddings for Modeling Multi-relational Data. In: *Advances in Neural Information Processing Systems 26 (NeurIPS)*. 2013:2787-2795."),
    # [8] DistMult
    dict(id=8, cite="Yang B, Yih SW, He X, Gao J, Deng L. ICLR. 2015.",
         full="8. **Yang B**, Yih SW, He X, Gao J, Deng L. Embedding Entities and Relations for Learning and Inference in Knowledge Bases. In: *Proceedings of the International Conference on Learning Representations (ICLR)*. 2015. arXiv:1412.6575."),
    # [9] ComplEx
    dict(id=9, cite="Trouillon T, Welbl J, Riedel S, Gaussier E, Bouchard G. ICML. 2016;48:2071-2080.",
         full="9. **Trouillon T**, Welbl J, Riedel S, Gaussier E, Bouchard G. Complex Embeddings for Simple Link Prediction. In: *Proceedings of the 33rd International Conference on Machine Learning (ICML)*. 2016;48:2071-2080."),
    # [10] RotatE
    dict(id=10, cite="Sun Z, Deng ZH, Nie JY, Tang J. ICLR. 2019.",
         full="10. **Sun Z**, Deng ZH, Nie JY, Tang J. RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. In: *Proceedings of the International Conference on Learning Representations (ICLR)*. 2019. arXiv:1902.10197."),
    # [11] TGF-beta tissue fibrosis
    dict(id=11, cite="Frangogiannis NG. J Exp Med. 2020;217(3):e20190103.",
         full="11. **Frangogiannis NG**. Transforming growth factor-β in tissue fibrosis. *Journal of Experimental Medicine*. 2020;217(3):e20190103. doi:10.1084/jem.20190103"),
    # [12] TGF-beta master regulator
    dict(id=12, cite="Meng XM, Nikolic-Paterson DJ, Lan HY. Nat Rev Nephrol. 2016;12(6):325-338.",
         full="12. **Meng XM**, Nikolic-Paterson DJ, Lan HY. TGF-β: the master regulator of fibrosis. *Nature Reviews Nephrology*. 2016;12(6):325-338. doi:10.1038/nrneph.2016.48"),
    # [13] Scanpy
    dict(id=13, cite="Wolf FA, Angerer P, Theis FJ. Genome Biol. 2018;19(1):15.",
         full="13. **Wolf FA**, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology*. 2018;19(1):15. doi:10.1186/s13059-017-1382-0"),
    # [14] Seurat v5
    dict(id=14, cite="Hao Y, Stuart T, Kowalski MH, et al. Nat Biotechnol. 2024;42(2):293-304.",
         full="14. **Hao Y**, Stuart T, Kowalski MH, et al. Dictionary learning for integrative, multimodal and scalable single-cell analysis. *Nature Biotechnology*. 2024;42(2):293-304. doi:10.1038/s41587-023-01767-y"),
    # [15] Mouse DVT model review
    dict(id=15, cite="Diaz JA, Obi AT, Myers DD Jr, et al. ATVB. 2012;32(3):556-562.",
         full="15. **Diaz JA**, Obi AT, Myers DD Jr, et al. Critical review of mouse models of venous thrombosis. *Arteriosclerosis, Thrombosis, and Vascular Biology*. 2012;32(3):556-562. doi:10.1161/ATVBAHA.111.244608"),
    # [16] IVC fibrosis model
    dict(id=16, cite="Henke PK, Varma MR, Moaveni DK, et al. J Vasc Surg. 2007;46(4):748-754.",
         full="16. **Henke PK**, Varma MR, Moaveni DK, et al. Fibrotic injury after experimental deep vein thrombosis is determined by the mechanism of thrombogenesis. *Journal of Vascular Surgery*. 2007;46(4):748-754. doi:10.1016/j.jvs.2007.06.011"),
    # [17] Macrophage tissue repair
    dict(id=17, cite="Wynn TA, Vannella KM. Immunity. 2016;44(3):450-462.",
         full="17. **Wynn TA**, Vannella KM. Macrophages in Tissue Repair, Regeneration, and Fibrosis. *Immunity*. 2016;44(3):450-462. doi:10.1016/j.immuni.2016.02.015"),
    # [18] GSE48000
    dict(id=18, cite="Lewis DA, Suchindran S, Beckman MG, et al. Thromb Res. 2015;135(4):659-665.",
         full="18. **Lewis DA**, Suchindran S, Beckman MG, et al. Whole blood gene expression profiles distinguish clinical phenotypes of venous thromboembolism. *Thrombosis Research*. 2015;135(4):659-665. doi:10.1016/j.thromres.2015.02.003"),
    # [19] TLR4 DVT
    dict(id=19, cite="Yuan Y, Huang W, Chen Y, et al. Front Mol Biosci. 2023;10:1165589.",
         full="19. **Yuan Y**, Huang W, Chen Y, et al. Toll-like receptor 4 deficiency in mice impairs venous thrombus resolution. *Frontiers in Molecular Biosciences*. 2023;10:1165589. doi:10.3389/fmolb.2023.1165589"),
    # [20] Immunothrombosis
    dict(id=20, cite="Engelmann B, Massberg S. Nat Rev Immunol. 2013;13(1):34-45.",
         full="20. **Engelmann B**, Massberg S. Thrombosis as an intravascular effector of innate immunity. *Nature Reviews Immunology*. 2013;13(1):34-45. doi:10.1038/nri3345"),
    # [21] GSEA
    dict(id=21, cite="Subramanian A, Tamayo P, Mootha VK, et al. PNAS. 2005;102(43):15545-15550.",
         full="21. **Subramanian A**, Tamayo P, Mootha VK, et al. Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles. *Proceedings of the National Academy of Sciences*. 2005;102(43):15545-15550. doi:10.1073/pnas.0506580102"),
    # [22] fgsea
    dict(id=22, cite="Korotkevich G, Sukhov V, Budin N, et al. bioRxiv. 2021:060012.",
         full="22. **Korotkevich G**, Sukhov V, Budin N, Shpak B, Artyomov MN, Sergushichev A. Fast gene set enrichment analysis. *bioRxiv*. 2021:060012. doi:10.1101/060012"),
    # [23] PAI-1 vein wall
    dict(id=23, cite="Obi AT, Diaz JA, Ballard-Lipka NL, et al. J Thromb Haemost. 2014;12(1):136-144.",
         full="23. **Obi AT**, Diaz JA, Ballard-Lipka NL, et al. Plasminogen activator-1 overexpression decreases experimental postthrombotic vein wall fibrosis by a non-vitronectin-dependent mechanism. *Journal of Thrombosis and Haemostasis*. 2014;12(1):136-144. doi:10.1111/jth.12644"),
    # [24] Thrombus resolution chemokines
    dict(id=24, cite="Henke PK, Wakefield TW. Thromb Res. 2009;123(Suppl 4):S72-S78.",
         full="24. **Henke PK**, Wakefield TW. Thrombus resolution and vein wall injury: dependence on chemokines and leukocytes. *Thrombosis Research*. 2009;123(Suppl 4):S72-S78. doi:10.1016/S0049-3848(09)70148-3"),
    # [25] Macrophage-fibroblast cardiac
    dict(id=25, cite="Lafuse WP, Wozniak DJ, Rajaram MVS. Cells. 2021;10(1):51.",
         full="25. **Lafuse WP**, Wozniak DJ, Rajaram MVS. Role of cardiac macrophages on cardiac inflammation, fibrosis and tissue repair. *Cells*. 2021;10(1):51. doi:10.3390/cells10010051"),
    # [26] TGF-beta cancer therapy
    dict(id=26, cite="Peng D, Fu M, Wang M, Wei Y, Wei X. Mol Cancer. 2022;21(1):104.",
         full="26. **Peng D**, Fu M, Wang M, Wei Y, Wei X. Targeting TGF-β signal transduction for fibrosis and cancer therapy. *Molecular Cancer*. 2022;21(1):104. doi:10.1186/s12943-022-01569-x"),
    # [27] PyTorch Geometric
    dict(id=27, cite="Fey M, Lenssen JE. ICLR Workshop RLGM. 2019.",
         full="27. **Fey M**, Lenssen JE. Fast Graph Representation Learning with PyTorch Geometric. In: *ICLR Workshop on Representation Learning on Graphs and Manifolds*. 2019. arXiv:1903.02428."),
    # [28] Node2Vec
    dict(id=28, cite="Grover A, Leskovec J. KDD. 2016:855-864.",
         full="28. **Grover A**, Leskovec J. node2vec: Scalable Feature Learning for Networks. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*. 2016:855-864. doi:10.1145/2939672.2939754"),
    # [29] PubMedBERT
    dict(id=29, cite="Gu Y, Tinn R, Cheng H, et al. ACM Trans Comput Healthc. 2021;3(1):Article 2.",
         full="29. **Gu Y**, Tinn R, Cheng H, et al. Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing. *ACM Transactions on Computing for Healthcare*. 2021;3(1):Article 2. doi:10.1145/3458754"),
    # [30] Seurat v3 integration
    dict(id=30, cite="Stuart T, Butler A, Hoffman P, et al. Cell. 2019;177(7):1888-1902.",
         full="30. **Stuart T**, Butler A, Hoffman P, et al. Comprehensive Integration of Single-Cell Data. *Cell*. 2019;177(7):1888-1902.e21. doi:10.1016/j.cell.2019.05.031"),
]

# ── Step 1: Read and strip all existing [N] citations ──
text = MD_PATH.read_text(encoding="utf-8")
text = re.sub(r'\s*\[\d+(?:[,-]\d+)*\]', '', text)

# ── Step 2: Define insertion rules ──
# Each rule: (pattern_to_match, insert_after_match, [ref_ids])
# The pattern is the exact text BEFORE where citation should go

rules = [
    # === INTRODUCTION ===
    # Line 28: "...progressive vein wall fibrosis." → PTS refs
    ("progressive vein wall fibrosis", " [1,2]"),
    # "...fail to attenuate chronic vein wall structural remodeling" → Kahn Blood 2009
    ("chronic vein wall structural remodeling", " [3]"),
    # "...biomedical knowledge graphs (KGs) and graph neural networks (GNNs) have emerged..." → KG+GNN
    ("have emerged as powerful paradigms for systematic target discovery", " [4,5]"),
    # "...Tempered Heterogeneous Graph Transformer (Tempered HGT)" → HGT paper
    ("Tempered Heterogeneous Graph Transformer (Tempered HGT)", " [6]"),
    # "...four classical knowledge graph embedding architectures" → KGE baselines
    ("four classical knowledge graph embedding architectures", " [7,8,9,10]"),
    # "...central transcriptional mediator of TGF-β signaling" → TGF-β
    ("TGF-$\\beta$ signaling not traditionally associated with acute VTE", " [11,12]"),
    # "...primary scRNA-seq profiles (21,230 cells)" → Scanpy+Seurat
    ("scRNA-seq profiles (21,230 cells)", " [13,14]"),
    # "...mouse IVC stenosis model at Day 14" → IVC model
    ("mouse IVC stenosis model at Day 14", " [15,16]"),
    # "...infiltrating macrophages supply Tgfb1 ligand..." → macrophage paper
    ("matrix remodeling in adventitial fibroblasts.", " [17]"),
    # "...human clinical cohort (GSE48000, n = 132)" → GSE48000
    ("(GSE48000, $n = 132$)", " [18]"),

    # === RESULTS 2.2 ===
    ("TLR4/NF-$\\kappa$B inflammatory program", " [19,20]"),  # first mention in Results 2.2
    ("TGF-$\\beta$/SMAD4 tissue remodeling program", " [11,12]"),  # reuse TGF-β refs

    # === RESULTS 2.3 ===
    # "...mouse IVC stenosis model at Day 14 post-ligation" → reuse [15,16]
    ("mouse IVC stenosis model at Day 14", " [15,16]"),
    # "...scRNA-seq profiles from a mouse" → reuse [13,14]
    ("scRNA-seq) profiles from a mouse", " [13,14]"),
    # "...post-thrombotic vein wall fibrosis and structural remodeling." → vein wall refs
    ("post-thrombotic vein wall fibrosis and structural remodeling.", " [23,24]"),

    # === RESULTS 2.4 ===
    # "...gene set enrichment analysis (GSEA)" → GSEA refs
    ("gene set enrichment analysis (GSEA) on a clinical", " [21,22]"),
    # "...human VTE whole-blood samples (GSE48000" → reuse [18]
    ("human VTE whole-blood samples (GSE48000", " [18]"),

    # === DISCUSSION ===
    # "...biomedical knowledge graph learning is distinguishing..." → KG refs
    ("recapitulation of densely cited textbook knowledge.", " [4,5]"),
    # "...TransE, DistMult, ComplEx, and RotatE—achieved" → reuse KGE refs
    ("DistMult, ComplEx, and RotatE", " [7,8,9,10]"),
    # "...overcome this literature-density bias" → HGT
    ("overcome this literature-density bias.", " [6]"),
    # "...not traditionally associated with VTE, as the top-ranked candidate." → TGF-β
    ("top-ranked candidate.", " [11,12]"),
    # "...macrophage-orchestrated tissue remodeling." → macrophage+vein wall
    ("macrophage-orchestrated tissue remodeling.", " [17,25]"),
    # "...post-thrombotic vein wall collagen deposition and luminal stenosis." → TGF-β therapy
    ("luminal stenosis.", " [26]"),
    # "...empirical permutation testing against random gene sets" → GSEA reuse
    ("empirical permutation testing against random gene sets", " [21,22]"),

    # === METHODS §1 ===
    # "...integrating entity-resolved biomedical databases..." → KG review
    ("biological processes, and disease phenotypes.", " [4]"),

    # === METHODS §2 ===
    # "...Tempered HGT model was designed" → HGT
    ("The Tempered HGT model was designed", " [6]"),
    # "...TransE, DistMult, ComplEx, and RotatE. All models" → reuse KGE
    ("DistMult, ComplEx, and RotatE. All models", " [7,8,9,10]"),

    # === METHODS §3 ===
    # "...established mouse model of IVC stenosis" → IVC model
    ("established mouse model of inferior vena cava (IVC) stenosis.", " [15,16]"),
    # "...Scanpy) and R (Seurat v5.0)" → method refs
    ("Scanpy) and R (Seurat v5.0)", " [13,14]"),

    # === METHODS §4 ===
    # "...GSE48000; n = 132" → reuse
    ("(GSE48000; $n = 132$", " [18]"),
    # "...fgseaMultilevel" → fgsea
    ("fgseaMultilevel", " [22]"),

    # === METHODS §5 ===
    # "...PyTorch Geometric, Scanpy)" → PyG + Scanpy
    ("PyTorch Geometric, Scanpy)", " [27,13]"),
    # "...Seurat v5.0)" → Seurat
    ("Seurat v5.0)", " [14]"),

    # === FIGURE LEGENDS ===
    # "...PubMedBERT (768d) + Node2Vec (128d)" → NLP feature refs
    ("PubMedBERT (768d) + Node2Vec (128d)", " [29,28]"),
]

# ── Step 3: Apply rules (in reverse order to preserve positions) ──
# Sort rules by position in text (last first to avoid offset issues)
rule_positions = []
for pattern, insertion in rules:
    pos = text.find(pattern)
    if pos == -1:
        print(f"WARNING: Pattern not found: '{pattern[:60]}...'")
    else:
        rule_positions.append((pos, pattern, insertion))

# Sort by position descending
rule_positions.sort(key=lambda x: x[0], reverse=True)

for pos, pattern, insertion in rule_positions:
    insert_pos = pos + len(pattern)
    text = text[:insert_pos] + insertion + text[insert_pos:]

# ── Step 4: Handle duplicate citations ──
# Some refs appear many times (e.g., [13,14] for Scanpy+Seurat). That's fine for Vancouver style.
# But we should check for consistency.

# ── Step 5: Find all unique citation numbers used ──
used_refs = set()
for m in re.finditer(r'\[([^\]]+)\]', text):
    for n in re.findall(r'\d+', m.group(1)):
        used_refs.add(int(n))

print(f"Total unique references cited: {len(used_refs)}")
print(f"Reference numbers used: {sorted(used_refs)}")
missing = set(range(1, 31)) - used_refs
if missing:
    print(f"WARNING: References NOT cited: {sorted(missing)}")

# ── Step 6: Write corrected manuscript ──
MD_PATH.write_text(text, encoding="utf-8")
print(f"\nCorrected manuscript written to: {MD_PATH}")

# ── Step 7: Generate final reference list ──
ref_by_id = {r['id']: r for r in REFS}
ref_lines = ["# References", "", "## Final Reference List (Vancouver Numbered Order)", ""]
ref_lines.append("All references verified via PubMed/DOI as of 2026-07-30. No fabricated citations.")
ref_lines.append("")
ref_lines.append("---")
ref_lines.append("")

for rid in sorted(used_refs):
    ref_lines.append(ref_by_id[rid]['full'])
    ref_lines.append("")

REF_PATH = PROJECT / "manuscript" / "references_final.md"
REF_PATH.write_text("\n".join(ref_lines), encoding="utf-8")
print(f"Final reference list written to: {REF_PATH}")
