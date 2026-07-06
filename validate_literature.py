#!/usr/bin/env python3
"""Literature novelty validator for hidden targets — uses KG topology + blacklist.

Memory-safe: processes one target at a time, no Neo4j required.
Classifies: novel_mechanism / underexplored / emerging / known_in_vte.
"""
import json, os, time, math
from datetime import datetime
from collections import defaultdict

def log(msg):
    print(msg, flush=True)

# ── Textbook VTE targets (from hidden_target_hunter.py) ──
TEXTBOOK_BLACKLIST = {
    'f2', 'prothrombin', 'thrombin', 'f5', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
    'f13', 'tissue factor', 'vwd', 'vwf', 'von willebrand', 'adamts13',
    'fxa', 'factor xa', 'fva', 'fvii', 'fvii1', 'fviii', 'fix', 'fxi', 'fxii',
    'protein c', 'protein s', 'antithrombin', 'serpine1', 'pai-1', 'tfp1',
    'thrombomodulin', 'epcr',
    'fibrinogen', 'fibrin', 'fga', 'fgb', 'fgg', 'plasminogen', 'plg', 'tpa',
    'plasmin', 'alpha-2-antiplasmin',
    'p-selectin', 'selectin', 'gpiib', 'gpiia', 'gpib', 'gpvi',
    'il-6', 'il-1', 'il-8', 'tnf', 'nf-kb', 'nfkb', 'tlr4', 'tlr2',
    'cox-2', 'il-10', 'il-12', 'il-4', 'il-17',
    'vegf', 'nos3', 'eno', 'endothelin',
    'factor v leiden', 'prothrombin 20210', 'mthfr', 'jak2',
    'hmgb1', 'padi4', 'albumin', 'glucose', 'cholesterol',
}

# ── VTE-relevant disease/process keywords ──
VTE_KEYWORDS = [
    'thrombosis', 'thromboembolism', 'dvt', 'deep vein', 'pulmonary embol',
    'venous', 'vte', 'coagulation', 'platelet', 'fibrin', 'endothelial',
    'vascul', 'stroke', 'myocardial', 'cardiovascular',
]

def is_blacklisted(name):
    n = name.lower().strip()
    for term in TEXTBOOK_BLACKLIST:
        if term in n:
            return True
    return False

def contains_vte_context(name):
    n = name.lower()
    for kw in VTE_KEYWORDS:
        if kw in n:
            return True
    return False

def classify_novelty(target_name, degree, neighbor_types, vte_neighbors,
                     anchor_hits, pathways, blacklisted):
    """Classify a hidden target's novelty level.

    Returns: (category, confidence, rationale)
    """
    if blacklisted:
        return 'textbook_known', 0.95, 'matches textbook VTE blacklist'

    # Novel mechanism: low degree (<50), not in blacklist, connects to cascade
    if degree < 50 and anchor_hits:
        vte_context = sum(1 for _, _, ctx in vte_neighbors if ctx)
        if vte_context <= 1:
            return 'novel_mechanism', 0.70, (
                f'low degree ({degree}), minimal VTE literature context ({vte_context} VTE neighbors), '
                f'connects to mechanism cascade via {"+".join(anchor_hits[:2])}'
            )
        else:
            return 'underexplored', 0.65, (
                f'low degree ({degree}), some VTE context ({vte_context} neighbors), '
                f'mechanism cascade connection via {"+".join(anchor_hits[:2])}'
            )

    # Underexplored: medium degree (50-200), connects to cascade
    if 50 <= degree < 200 and anchor_hits:
        return 'underexplored', 0.60, (
            f'medium degree ({degree}), connects to mechanism cascade via {"+".join(anchor_hits[:2])}'
        )

    # Emerging: medium-high degree, several VTE connections
    if degree >= 200:
        return 'emerging', 0.55, (
            f'high degree ({degree}), may have broader biological role beyond VTE'
        )

    return 'undetermined', 0.30, f'degree={degree}, insufficient classification features'

def main():
    t0 = time.time()
    log("=" * 60)
    log(f"Literature Novelty Validator   {datetime.now():%H:%M:%S}")
    log("=" * 60)

    # ── Load KG data ──
    log("\n[1/3] Loading KG topology...")
    import torch
    data = torch.load("data/processed/heterodata.pt", weights_only=False)

    # Node names
    node_names = {}
    for nt in data.node_types:
        if hasattr(data[nt], 'name'):
            for idx, name in enumerate(data[nt].name):
                node_names[(nt, idx)] = str(name)

    # Degree computation
    degree = defaultdict(int)
    for et in data.edge_types:
        ei = data[et].edge_index
        for j in range(ei.shape[1]):
            src, dst = int(ei[0,j]), int(ei[1,j])
            degree[(et[0], src)] += 1
            degree[(et[2], dst)] += 1

    # Per-node neighbor types (plain dict of lists, memory safe)
    neighbor_types = defaultdict(list)
    for et in data.edge_types:
        ei = data[et].edge_index
        for j in range(ei.shape[1]):
            src, dst = int(ei[0,j]), int(ei[1,j])
            neighbor_types[(et[0], src)].append((et[2], dst, et[1]))
            neighbor_types[(et[2], dst)].append((et[0], src, et[1]))

    # ── Load subgraph data ──
    log("\n[2/3] Loading target subgraphs...")
    with open("figures/pca_hidden/subgraphs/_summary.json", encoding='utf-8') as f:
        summary = json.load(f)
    with open("figures/pca_hidden/hidden_top15.json", encoding='utf-8') as f:
        hidden15 = json.load(f)

    targets_data = summary.get('targets', [])
    targets_lookup = {t['target']: t for t in hidden15}

    # ── Validate each target ──
    log("\n[3/3] Validating novelty...")
    results = []

    for tgt in targets_data:
        name = tgt['name']
        log(f"\n  {'='*50}")
        log(f"  Target: {name}")
        log(f"  {'='*50}")

        # Load subgraph
        json_path = tgt['output_file']
        with open(json_path, encoding='utf-8') as f:
            sg = json.load(f)

        # Find node in KG — sort by degree descending, prefer exact match
        found_node = None
        candidates = []
        for (nt, idx), nn in node_names.items():
            nl = nn.lower().strip()
            search_lower = name.lower().strip()
            # Exact match = highest priority
            if nl == search_lower:
                d = degree.get((nt, idx), 0)
                candidates.append((0, d, nt, idx, nn))  # priority 0
            elif search_lower in nl and nl != search_lower:
                d = degree.get((nt, idx), 0)
                # Prefer protein/gene/drug types, suppress Article/Concept
                type_priority = 0 if nt in ('Protein', 'Gene', 'Drug', 'Cytokine') else 1
                candidates.append((1 + type_priority, d, nt, idx, nn))

        if candidates:
            # Sort: (priority, -degree) -> lowest priority, highest degree first
            candidates.sort(key=lambda x: (x[0], -x[1]))
            _, _, nt, idx, nn = candidates[0]
            found_node = (nt, idx, nn)

        if found_node is None:
            log(f"  NOT FOUND in KG node_names")
            results.append({'target': name, 'status': 'not_found_in_kg'})
            continue

        nt, idx, full_name = found_node
        d = degree.get((nt, idx), 0)
        nbrs = neighbor_types.get((nt, idx), [])

        # Count VTE-context neighbors
        vte_nbrs = []
        for nbr_type, nbr_idx, nbr_rel in nbrs[:200]:
            if (nbr_type, nbr_idx) in node_names:
                nbr_name = node_names[(nbr_type, nbr_idx)]
                vte_ctx = contains_vte_context(nbr_name)
                if vte_ctx:
                    vte_nbrs.append((nbr_type, nbr_name[:60], True))
                elif nbr_type in ('Disease', 'Process', 'Pathway'):
                    vte_ctx2 = contains_vte_context(str(nbr_name))
                    if vte_ctx2:
                        vte_nbrs.append((nbr_type, nbr_name[:60], True))

        # Count neighbor types
        nbr_type_counts = defaultdict(int)
        for nbr_type, _, _ in nbrs:
            nbr_type_counts[nbr_type] += 1

        # Anchor hits from cascade mapping
        cascade_map = tgt.get('cascade_mapping', [])
        anchor_hits = [c for c in cascade_map if c not in ('Fibrosis', 'Inflammat')]

        # Pathways from hidden_top15
        pathways = targets_lookup.get(name, {}).get('pathways', [])

        # Check blacklist
        bl = is_blacklisted(name) or is_blacklisted(full_name)

        # Classify novelty
        category, confidence, rationale = classify_novelty(
            name, d, nbr_type_counts, vte_nbrs,
            anchor_hits, pathways, bl
        )

        # Additional: check if target name contains VTE keywords
        name_has_vte = contains_vte_context(name) or contains_vte_context(full_name)

        result = {
            'target': name,
            'full_name': full_name[:80],
            'target_type': nt,
            'degree': d,
            'neighbor_type_distribution': dict(nbr_type_counts),
            'vte_context_neighbors': min(len(vte_nbrs), 20),
            'vte_neighbor_sample': [(t, n) for t, n, _ in vte_nbrs[:5]],
            'blacklisted': bl,
            'name_contains_vte_keyword': name_has_vte,
            'cascade_mapping': cascade_map,
            'anchor_hits': anchor_hits,
            'shortest_path_hops': tgt.get('shortest_hops'),
            'novelty_category': category,
            'novelty_confidence': confidence,
            'novelty_rationale': rationale,
            'gnn_score': targets_lookup.get(name, {}).get('gnn_score', 0),
            'discovery_score': targets_lookup.get(name, {}).get('discovery_score', 0),
        }

        log(f"  Node: {full_name[:80]}")
        log(f"  Type: {nt} | Degree: {d} | VTE neighbors: {len(vte_nbrs)}")
        log(f"  Neighbor types: {dict(nbr_type_counts)}")
        log(f"  Blacklisted: {bl} | Name has VTE: {name_has_vte}")
        log(f"  Cascade: {cascade_map} | Anchors: {anchor_hits}")
        log(f"  -> NOVELTY: {category} (confidence={confidence:.2f})")
        log(f"  -> {rationale}")

        if vte_nbrs:
            log(f"  VTE-adjacent nodes:")
            for item in vte_nbrs[:5]:
                tn, nn = item[0], item[1]
                log(f"    [{tn}] {nn}")

        results.append(result)

    # ── Summary ──
    log(f"\n{'='*60}")
    log(f"NOVELTY SUMMARY")
    log(f"{'='*60}")

    categories = defaultdict(list)
    for r in results:
        categories[r.get('novelty_category', 'unknown')].append(r['target'])

    for cat in ['novel_mechanism', 'underexplored', 'emerging', 'known_in_vte', 'undetermined']:
        if cat in categories:
            targets_in_cat = categories[cat]
            log(f"  {cat}: {len(targets_in_cat)} — {', '.join(targets_in_cat)}")

    # ── Export ──
    output = {
        'method': 'KG topology + blacklist-based novelty classification',
        'note': 'Neo4j offline — using cached KG topology, no PMID enrichment',
        'timestamp': datetime.now().isoformat(),
        'targets': results,
        'summary': {cat: len(items) for cat, items in categories.items()},
    }

    os.makedirs("figures/pca_hidden", exist_ok=True)
    out_path = "figures/pca_hidden/literature_validation.json"
    with open(out_path, "w", encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    t = time.time() - t0
    log(f"\nDone in {t:.1f}s")
    log(f"Results: {out_path}")

    # ── Highlight: most novel targets ──
    novel = [r for r in results if r['novelty_category'] == 'novel_mechanism']
    underex = [r for r in results if r['novelty_category'] == 'underexplored']
    log(f"\n*** Key Finding ***")
    if novel:
        log(f"  NOVEL MECHANISM: {', '.join(r['target'] for r in novel)}")
    if underex:
        log(f"  UNDEREXPLORED: {', '.join(r['target'] for r in underex)}")

if __name__ == "__main__":
    main()
