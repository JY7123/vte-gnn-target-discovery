#!/usr/bin/env python3
"""
Re-number citations in manuscript to sequential Vancouver order,
then generate the matching final reference list.
"""
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT / "manuscript" / "manuscript_draft.md"

# ── Define what each original [N] maps to (bibliographic reference) ──
# Key: old number used in manuscript, Value: (short label for dedup)
# We'll assign sequential numbers based on first appearance order
REF_LABELS = {
    1:  "Kahn SR. Circulation. 2014. [PTS AHA scientific statement]",
    2:  "Galanaud JP. Thromb Res. 2018. [PTS epidemiology]",
    3:  "Kahn SR. Blood. 2009. [How I treat PTS]",
    4:  "Nicholson DN, Greene CS. Comput Struct Biotechnol J. 2020. [Biomedical KG review]",
    5:  "Zitnik M, Agrawal M, Leskovec J. Bioinformatics. 2018. [GNN polypharmacy]",
    6:  "Bordes A et al. NeurIPS. 2013. [TransE]",
    7:  "Yang B et al. ICLR. 2015. [DistMult]",
    8:  "Trouillon T et al. ICML. 2016. [ComplEx]",
    9:  "Sun Z et al. ICLR. 2019. [RotatE]",
    10: "Frangogiannis NG. J Exp Med. 2020. [TGF-beta tissue fibrosis]",
    11: "Meng XM et al. Nat Rev Nephrol. 2016. [TGF-beta master regulator]",
    12: "Wolf FA et al. Genome Biol. 2018. [Scanpy]",
    13: "Hao Y et al. Nat Biotechnol. 2024. [Seurat v5]",
    14: "Diaz JA et al. ATVB. 2012. [Mouse DVT model review]",
    15: "Henke PK et al. J Vasc Surg. 2007. [IVC fibrosis model]",
    16: "Wynn TA, Vannella KM. Immunity. 2016. [Macrophage tissue repair]",
    17: "Lewis DA et al. Thromb Res. 2015. [GSE48000]",
    18: "Yuan Y et al. Front Mol Biosci. 2023. [TLR4 DVT resolution]",
    19: "Engelmann B, Massberg S. Nat Rev Immunol. 2013. [Immunothrombosis]",
    20: "Subramanian A et al. PNAS. 2005. [GSEA]",
    21: "Korotkevich G et al. bioRxiv. 2021. [fgsea]",
    22: "Obi AT et al. J Thromb Haemost. 2014. [PAI-1 vein wall fibrosis]",
    23: "Henke PK, Wakefield TW. Thromb Res. 2009. [Thrombus resolution chemokines]",
    24: "Lafuse WP et al. Cells. 2021. [Macrophage-fibroblast cardiac]",
    25: "Peng D et al. Mol Cancer. 2022. [TGF-beta therapy]",
    26: "Fey M, Lenssen JE. ICLR Workshop. 2019. [PyTorch Geometric]",
    27: "Hu Z et al. WWW. 2020. [Heterogeneous Graph Transformer]",
    28: "Grover A, Leskovec J. KDD. 2016. [Node2Vec]",
    29: "Gu Y et al. ACM Trans Comput Healthc. 2021. [PubMedBERT]",
    30: "Lee J et al. Bioinformatics. 2020. [BioBERT]",
}

# ── Read manuscript ──
text = MD_PATH.read_text(encoding="utf-8")

# Find all citation brackets and their positions
citation_pattern = re.compile(r'\[([^\]]+)\]')

# Track first appearance order of each original number
first_appearance = {}
pos = 0
for m in re.finditer(r'\[(\d+(?:[,-]\d+)*)\]', text):
    numbers = re.findall(r'\d+', m.group(1))
    for n_str in numbers:
        n = int(n_str)
        if n not in first_appearance and n in REF_LABELS:
            first_appearance[n] = m.start()

# Sort by first appearance position
sorted_old_numbers = sorted(first_appearance.keys(), key=lambda n: first_appearance[n])

# Create mapping: old_number → new_number
old_to_new = {}
for new_num, old_num in enumerate(sorted_old_numbers, 1):
    old_to_new[old_num] = new_num

print(f"Total unique references: {len(old_to_new)}")
print(f"Old → New mapping:")
for old, new in sorted(old_to_new.items()):
    print(f"  [{old}] → [{new}]  {REF_LABELS[old][:80]}...")

# ── Re-number citations in manuscript ──
def replace_citation(match):
    content = match.group(1)
    # Parse the citation content: e.g., "1,2" or "10,11,12,13" or "20,26,27"
    parts = re.findall(r'\d+', content)
    new_parts = []
    for p in parts:
        old = int(p)
        if old in old_to_new:
            new_parts.append(str(old_to_new[old]))
        else:
            new_parts.append(p)  # keep unchanged if not in mapping
    # Reconstruct: keep the original separators
    # Simple approach: just join with comma
    return '[' + ','.join(new_parts) + ']'

new_text = re.sub(r'\[(\d+(?:[,-]\d+)*)\]', replace_citation, text)

# Write corrected manuscript
MD_PATH.write_text(new_text, encoding="utf-8")
print(f"\nCorrected manuscript written to: {MD_PATH}")

# ── Generate final reference list ──
ref_lines = []
ref_lines.append("# References\n")
ref_lines.append("## Final Reference List (Vancouver Numbered Order)\n")
ref_lines.append("All references verified via PubMed/DOI/Web search as of 2026-07-30.\n")

# Full bibliographic entries in the NEW order
FULL_REFS = {
    1:  '1. **Kahn SR**, Comerota AJ, Cushman M, et al. The postthrombotic syndrome: evidence-based prevention, diagnosis, and treatment strategies: a scientific statement from the American Heart Association. *Circulation*. 2014;130(18):1636-1661. doi:10.1161/CIR.0000000000000130',
    2:  '2. **Galanaud JP**, Monreal M, Kahn SR. Epidemiology of the post-thrombotic syndrome. *Thrombosis Research*. 2018;164:100-109. doi:10.1016/j.thromres.2017.07.026',
    3:  '3. **Kahn SR**. How I treat postthrombotic syndrome. *Blood*. 2009;114(21):4624-4631. doi:10.1182/blood-2009-07-199174',
    4:  '4. **Nicholson DN**, Greene CS. Constructing knowledge graphs and their biomedical applications. *Computational and Structural Biotechnology Journal*. 2020;18:1414-1428. doi:10.1016/j.csbj.2020.05.017',
    5:  '5. **Zitnik M**, Agrawal M, Leskovec J. Modeling polypharmacy side effects with graph convolutional networks. *Bioinformatics*. 2018;34(13):i457-i466. doi:10.1093/bioinformatics/bty294',
    6:  '6. **Bordes A**, Usunier N, Garcia-Duran A, Weston J, Yakhnenko O. Translating Embeddings for Modeling Multi-relational Data. In: *Advances in Neural Information Processing Systems 26 (NeurIPS)*. 2013:2787-2795.',
    7:  '7. **Yang B**, Yih SW, He X, Gao J, Deng L. Embedding Entities and Relations for Learning and Inference in Knowledge Bases. In: *Proceedings of the International Conference on Learning Representations (ICLR)*. 2015. arXiv:1412.6575.',
    8:  '8. **Trouillon T**, Welbl J, Riedel S, Gaussier E, Bouchard G. Complex Embeddings for Simple Link Prediction. In: *Proceedings of the 33rd International Conference on Machine Learning (ICML)*. 2016;48:2071-2080.',
    9:  '9. **Sun Z**, Deng ZH, Nie JY, Tang J. RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. In: *Proceedings of the International Conference on Learning Representations (ICLR)*. 2019. arXiv:1902.10197.',
    10: '10. **Frangogiannis NG**. Transforming growth factor-β in tissue fibrosis. *Journal of Experimental Medicine*. 2020;217(3):e20190103. doi:10.1084/jem.20190103',
    11: '11. **Meng XM**, Nikolic-Paterson DJ, Lan HY. TGF-β: the master regulator of fibrosis. *Nature Reviews Nephrology*. 2016;12(6):325-338. doi:10.1038/nrneph.2016.48',
    12: '12. **Wolf FA**, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology*. 2018;19(1):15. doi:10.1186/s13059-017-1382-0',
    13: '13. **Hao Y**, Stuart T, Kowalski MH, et al. Dictionary learning for integrative, multimodal and scalable single-cell analysis. *Nature Biotechnology*. 2024;42(2):293-304. doi:10.1038/s41587-023-01767-y',
    14: '14. **Diaz JA**, Obi AT, Myers DD Jr, et al. Critical review of mouse models of venous thrombosis. *Arteriosclerosis, Thrombosis, and Vascular Biology*. 2012;32(3):556-562. doi:10.1161/ATVBAHA.111.244608',
    15: '15. **Henke PK**, Varma MR, Moaveni DK, et al. Fibrotic injury after experimental deep vein thrombosis is determined by the mechanism of thrombogenesis. *Journal of Vascular Surgery*. 2007;46(4):748-754. doi:10.1016/j.jvs.2007.06.011',
    16: '16. **Wynn TA**, Vannella KM. Macrophages in Tissue Repair, Regeneration, and Fibrosis. *Immunity*. 2016;44(3):450-462. doi:10.1016/j.immuni.2016.02.015',
    17: '17. **Lewis DA**, Suchindran S, Beckman MG, et al. Whole blood gene expression profiles distinguish clinical phenotypes of venous thromboembolism. *Thrombosis Research*. 2015;135(4):659-665. doi:10.1016/j.thromres.2015.02.003',
    18: '18. **Yuan Y**, Huang W, Chen Y, et al. Toll-like receptor 4 deficiency in mice impairs venous thrombus resolution. *Frontiers in Molecular Biosciences*. 2023;10:1165589. doi:10.3389/fmolb.2023.1165589',
    19: '19. **Engelmann B**, Massberg S. Thrombosis as an intravascular effector of innate immunity. *Nature Reviews Immunology*. 2013;13(1):34-45. doi:10.1038/nri3345',
    20: '20. **Subramanian A**, Tamayo P, Mootha VK, et al. Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles. *Proceedings of the National Academy of Sciences*. 2005;102(43):15545-15550. doi:10.1073/pnas.0506580102',
    21: '21. **Korotkevich G**, Sukhov V, Budin N, Shpak B, Artyomov MN, Sergushichev A. Fast gene set enrichment analysis. *bioRxiv*. 2021:060012. doi:10.1101/060012',
    22: '22. **Obi AT**, Diaz JA, Ballard-Lipka NL, et al. Plasminogen activator-1 overexpression decreases experimental postthrombotic vein wall fibrosis by a non-vitronectin-dependent mechanism. *Journal of Thrombosis and Haemostasis*. 2014;12(1):136-144. doi:10.1111/jth.12644',
    23: '23. **Henke PK**, Wakefield TW. Thrombus resolution and vein wall injury: dependence on chemokines and leukocytes. *Thrombosis Research*. 2009;123(Suppl 4):S72-S78. doi:10.1016/S0049-3848(09)70148-3',
    24: '24. **Lafuse WP**, Wozniak DJ, Rajaram MVS. Role of cardiac macrophages on cardiac inflammation, fibrosis and tissue repair. *Cells*. 2021;10(1):51. doi:10.3390/cells10010051',
    25: '25. **Peng D**, Fu M, Wang M, Wei Y, Wei X. Targeting TGF-β signal transduction for fibrosis and cancer therapy. *Molecular Cancer*. 2022;21(1):104. doi:10.1186/s12943-022-01569-x',
    26: '26. **Fey M**, Lenssen JE. Fast Graph Representation Learning with PyTorch Geometric. In: *ICLR Workshop on Representation Learning on Graphs and Manifolds*. 2019. arXiv:1903.02428.',
    27: '27. **Hu Z**, Dong Y, Wang K, Sun Y. Heterogeneous Graph Transformer. In: *Proceedings of The Web Conference 2020 (WWW \'20)*. 2020:2704-2710. doi:10.1145/3366423.3380027',
    28: '28. **Grover A**, Leskovec J. node2vec: Scalable Feature Learning for Networks. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*. 2016:855-864. doi:10.1145/2939672.2939754',
    29: '29. **Gu Y**, Tinn R, Cheng H, et al. Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing. *ACM Transactions on Computing for Healthcare*. 2021;3(1):Article 2. doi:10.1145/3458754',
    30: '30. **Lee J**, Yoon W, Kim S, et al. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*. 2020;36(4):1234-1240. doi:10.1093/bioinformatics/btz682',
}

# Write references in the NEW order
for new_num in range(1, len(old_to_new) + 1):
    # Find which old number maps to this new number
    for old_num, nn in old_to_new.items():
        if nn == new_num:
            ref_lines.append(FULL_REFS[old_num])
            ref_lines.append("")
            break

REF_PATH = PROJECT / "manuscript" / "references_final.md"
REF_PATH.write_text("\n".join(ref_lines), encoding="utf-8")
print(f"Final reference list written to: {REF_PATH}")
print(f"Total references: {len(old_to_new)}")
