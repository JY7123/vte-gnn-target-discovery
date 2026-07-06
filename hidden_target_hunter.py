#!/usr/bin/env python3
"""Hidden Target Hunter: discovery score ranking + degree penalty + mechanism anchoring.

Filters out textbook VTE targets, rewards low-degree high-signal nodes,
enforces 2-3 hop connectivity to FUT8/Lgals3/CD44 mechanism axis.
"""
import torch, json, time, math, os
from collections import defaultdict, deque
from datetime import datetime

# ── Textbook VTE Blacklist ──
TEXTBOOK_BLACKLIST = {
    # Coagulation cascade
    'f2', 'prothrombin', 'thrombin', 'f5', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
    'f13', 'tf', 'tissue factor', 'vwd', 'vwf', 'von willebrand', 'adamts13',
    'fxa', 'factor xa', 'fva', 'fvii', 'fvii1', 'fviii', 'fix', 'fxi', 'fxii',
    # Anticoagulation
    'protein c', 'protein s', 'antithrombin', 'serpine1', 'pai-1', 'tfp1',
    'thrombomodulin', 'epcr',
    # Fibrinolysis
    'fibrinogen', 'fibrin', 'fga', 'fgb', 'fgg', 'plasminogen', 'plg', 'tpa',
    'plasmin', 'alpha-2-antiplasmin',
    # Platelets
    'p-selectin', 'selectin', 'gpiib', 'gpiia', 'gpib', 'gpvi', 'gpia', 'collagen receptor',
    # Inflammation hubs
    'il-6', 'il-1', 'il-8', 'tnf', 'nf-kb', 'nfkb', 'tlr4', 'tlr2',
    'cox-2', 'il-10', 'il-12', 'il-4', 'il-17', 'il-18', 'il-1b',
    # Endothelial
    'vegf', 'nos3', 'eno', 'endothelin',
    # Known VTE genes
    'factor v leiden', 'prothrombin 20210', 'mthfr', 'jak2',
    # Falsified targets (Project 1)
    'hmgb1', 'padi4',
    # Common proteins/metabolites that are too broad
    'albumin', 'glucose', 'cholesterol', 'triglyceride',
}

# ── Core Mechanism Anchors ──
CORE_ANCHORS = {
    'fut8', 'lgals3', 'galectin-3', 'cd44', 'itgb1', 'itga5', 'itgav',
    'mapk1', 'mapk3', 'rhoa', 'rock1', 'rock2', 'nfkb1', 'rela',
}

def log(msg):
    print(msg, flush=True)

def is_blacklisted(name):
    n = name.lower()
    for term in TEXTBOOK_BLACKLIST:
        if term in n:
            return True
    return False

def is_anchor(name):
    n = name.lower()
    for a in CORE_ANCHORS:
        if a in n:
            return True
    return False

def resolve_anchor_name(name):
    """Extract clean gene/protein name from verbose Entity descriptions."""
    known = {
        'rhoa': 'RHOA', 'rock1': 'ROCK1', 'rock2': 'ROCK2',
        'mapk1': 'MAPK1', 'mapk3': 'MAPK3', 'nfkb1': 'NFKB1', 'rela': 'RELA',
        'cd44': 'CD44', 'itgb1': 'ITGB1', 'itga5': 'ITGA5',
        'fut8': 'FUT8', 'lgals3': 'LGALS3',
    }
    n = name.lower()
    for k, v in known.items():
        if k in n:
            return v
    return name[:30]

def main():
    t0 = time.time()
    log("=" * 60)
    log("HIDDEN TARGET HUNTER")
    log(f"{datetime.now():%H:%M:%S}")
    log("=" * 60)

    # ── Load ──
    log("\n[1/5] Loading data + v3 model...")
    data = torch.load("data/processed/heterodata.pt", weights_only=False)
    torch.manual_seed(42)
    for nt in data.node_types:
        data[nt].x = torch.randn(data[nt].num_nodes, 128) * 0.1

    ets = [(et, data[et].edge_index.shape[1]) for et in data.edge_types if et[1] != 'MENTIONED_IN']
    ets.sort(key=lambda x: -x[1])
    top20 = [et for et, n in ets[:20]]

    temp_init = {f'{s}__{r}__{d}': 1.0 for s, r, d in top20}
    from models.tempered_hgt import TemperedHGT
    model = TemperedHGT(
        in_channels={nt: 128 for nt in data.node_types},
        hidden_channels=128, out_channels=128, num_heads=4, num_layers=2,
        meta_relations=top20, temperature_init=temp_init,
    )
    ckpt = torch.load("checkpoints/overnight/checkpoint_epoch_53.pt", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    # ── Forward pass ──
    log("\n[2/5] Scoring all edges...")
    target_ets = [et for et in top20 if et[2] == 'Disease' and et[0] in ('Gene', 'Protein', 'Drug', 'Cytokine')]
    ei_dict = {et: data[et].edge_index for et in top20}
    x_dict = {nt: data[nt].x for nt in data.node_types}
    with torch.no_grad():
        z = model(x_dict, ei_dict, cos_decay=0.0)

    # Collect predictions with metadata
    predictions = []
    for et in target_ets:
        src_t, rel, dst_t = et
        ei = data[et].edge_index
        with torch.no_grad():
            scores = model.decode(z, ei, src_t, dst_t)
        sn = data[src_t].name if hasattr(data[src_t], 'name') else []
        dn = data[dst_t].name if hasattr(data[dst_t], 'name') else []
        for i in range(ei.shape[1]):
            s_idx = int(ei[0, i])
            name = sn[s_idx] if s_idx < len(sn) else str(s_idx)
            predictions.append({
                'name': name, 'src_type': src_t, 'relation': rel,
                'disease': dn[int(ei[1,i])] if int(ei[1,i]) < len(dn) else str(int(ei[1,i])),
                'score': float(scores[i]),
                'src_idx': s_idx,
            })

    # ── Degree computation ──
    log("\n[3/5] Computing node degrees + discovery scores...")
    degree = defaultdict(int)
    for et in data.edge_types:
        ei = data[et].edge_index
        for j in range(ei.shape[1]):
            degree[(et[0], int(ei[0,j]))] += 1
            degree[(et[2], int(ei[1,j]))] += 1

    # ── Build bidirectional adjacency for BFS ──
    log("Building adjacency for mechanism path tracing...")
    adj = defaultdict(list)
    for et in data.edge_types:
        src_t, rel, dst_t = et
        ei = data[et].edge_index
        for j in range(ei.shape[1]):
            s, d = int(ei[0,j]), int(ei[1,j])
            adj[(src_t, s)].append((dst_t, d, rel))
            adj[(dst_t, d)].append((src_t, s, f"rev_{rel}"))

    # ── Filter: blacklist removal + discovery score + anchor BFS ──
    log("\n[4/5] Hunting hidden targets...")
    hidden = []
    for p in predictions:
        name = p['name']
        # Skip blacklisted textbook targets
        if is_blacklisted(name):
            continue
        # Skip anchor genes themselves
        if is_anchor(name):
            continue

        # Compute discovery score with degree penalty
        d = degree.get((p['src_type'], p['src_idx']), 1)
        discovery_score = p['score'] / math.log(d + 1)

        # BFS to core anchors (max 3 hops)
        start = (p['src_type'], p['src_idx'])
        visited = {start: 0}
        q = deque([start])
        anchor_hit = None
        anchor_path = None
        while q and anchor_hit is None:
            cur = q.popleft()
            dist = visited[cur]
            if dist >= 3:
                continue
            for neighbor in adj.get(cur, []):
                nt_type, nt_idx, nt_rel = neighbor
                nkey = (nt_type, nt_idx)
                if nkey in visited:
                    continue
                visited[nkey] = dist + 1
                q.append(nkey)
                # Check if this neighbor is an anchor
                nn = ""
                if hasattr(data[nt_type], 'name') and nt_idx < len(data[nt_type].name):
                    nn = data[nt_type].name[nt_idx]
                if is_anchor(nn):
                    anchor_hit = resolve_anchor_name(nn)
                    anchor_path = dist + 1
                    break

        if anchor_hit is not None:
            p['discovery_score'] = round(discovery_score, 3)
            p['degree'] = d
            p['anchor_gene'] = anchor_hit
            p['anchor_hops'] = anchor_path
            hidden.append(p)

    # Sort by discovery score
    hidden.sort(key=lambda p: -p['discovery_score'])

    # Deduplicate: keep best discovery score per target name
    seen = {}
    deduped = []
    for p in hidden:
        key = p['name'].lower().strip()
        if key in seen:
            if p['discovery_score'] > seen[key].get('discovery_score', 0):
                seen[key] = p
        else:
            seen[key] = p
            deduped.append(p)
    hidden = sorted(deduped, key=lambda p: -p['discovery_score'])
    log(f"  After dedup: {len(hidden)} unique targets")
    log(f"  {len(hidden)} hidden targets connected to core mechanism anchors")

    # ── Export ──
    log("\n[5/5] Exporting hidden targets...")
    os.makedirs("figures/hidden_targets", exist_ok=True)

    # Top-30
    print(f"\n{'='*90}")
    print("TOP-30 HIDDEN TARGETS (blacklist-filtered, degree-penalized, anchor-connected)")
    print(f"{'='*90}")
    print(f"{'#':<3} {'Target':<48} {'Type':<8} {'GNN':<7} {'Deg':<5} {'DiscScore':<10} {'Anchor':<15} {'Hops'}")
    print("-"*90)

    results = []
    for i, p in enumerate(hidden[:30]):
        s = p['discovery_score']
        marker = " ⚠" if s < 0 else ""
        print(f"{i+1:<3} {p['name'][:46]:<48} {p['src_type']:<8} {p['score']:<7.2f} "
              f"{p['degree']:<5} {s:<10.3f} {p['anchor_gene'][:13]:<15} {p['anchor_hops']}{marker}")
        results.append({
            'rank': i+1, 'target': p['name'], 'type': p['src_type'],
            'gnn_score': round(p['score'], 2),
            'degree': p['degree'],
            'discovery_score': s,
            'anchor_gene': p['anchor_gene'],
            'anchor_hops': p['anchor_hops'],
            'disease': p['disease'],
            'relation': p['relation'],
        })

    # Pathway classification for top hidden targets
    PATHWAY_TERMS = {
        'Inflammation': ['inflamm', 'cytokine', 'il-', 'tnf', 'nf-kb', 'tlr', 'chemokine'],
        'Fibrosis': ['fibrosis', 'fibrotic', 'tgf', 'sma', 'collagen', 'mmp', 'timp', 'ctgf'],
        'Metabolism': ['metabol', 'lipid', 'glucose', 'insulin', 'cholesterol', 'fatty acid'],
        'Epigenetics': ['methyl', 'acetyl', 'hdac', 'sirt', 'mir', 'lncrna', 'histone'],
        'Oxidative Stress': ['oxid', 'ros', 'nadph', 'nox', 'sod', 'catalase', 'glutathione'],
        'Autophagy/Apoptosis': ['autophagy', 'apoptosis', 'caspase', 'bcl', 'bax', 'beclin'],
    }
    for p in hidden[:30]:
        n = p['name'].lower()
        p['novel_pathways'] = [k for k, terms in PATHWAY_TERMS.items() if any(t in n for t in terms)]

    # Save
    with open("figures/hidden_targets/hidden_top30.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Novelty breakdown
    novel_pathways = defaultdict(int)
    for p in hidden[:30]:
        for pw in p.get('novel_pathways', p.get('pathways', [])):
            if pw not in ('Coagulation', 'Inflammation', 'Fibrosis', 'Adhesion'):
                novel_pathways[pw] += 1

    if novel_pathways:
        print(f"\n--- Novel Mechanism Categories (beyond coagulation/inflammation) ---")
        for pw, cnt in sorted(novel_pathways.items(), key=lambda x: -x[1]):
            print(f"  {pw}: {cnt} targets")

    print(f"\nSaved: figures/hidden_targets/hidden_top30.json")
    print(f"Time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
