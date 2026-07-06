#!/usr/bin/env python3
"""Attention-weight mechanism subgraphs for top-5 hidden targets — Figure 4 data.

Uses native HGT attention weights (NOT gradient-based GNNExplainer) extracted
via pure forward pass — memory-safe, O(1) overhead, no backward graph retained.
"""
import torch, json, os, time
from collections import defaultdict, deque
from datetime import datetime

def log(msg):
    print(msg, flush=True)

t0 = time.time()
log("=" * 60)
log("Attention-Weight Mechanism Subgraphs — Top-5 Hidden Targets")
log(f"{datetime.now():%H:%M:%S}")
log("=" * 60)

# ── Load data & model ──────────────────────────────────────────────────────
log("\n[1/5] Loading data + PCA model...")
data = torch.load("data/processed/heterodata.pt", weights_only=False)
feats = torch.load("checkpoints/pca_features/features_128d.pt", weights_only=False)
for nt in data.node_types:
    data[nt].x = feats[nt]

ets = [(et, data[et].edge_index.shape[1]) for et in data.edge_types if et[1] != 'MENTIONED_IN']
ets.sort(key=lambda x: -x[1])
top20 = [et for et, n in ets[:20]]
log(f"  Top-20 edge types: {len(top20)}")

temp_init = {f'{s}__{r}__{d}': 1.0 for s, r, d in top20}
from models.tempered_hgt import TemperedHGT
model = TemperedHGT(
    in_channels={nt: 128 for nt in data.node_types},
    hidden_channels=128, out_channels=128, num_heads=4, num_layers=2,
    meta_relations=top20, temperature_init=temp_init,
)
with open("checkpoints/pca_features/summary.json") as f:
    train_summary = json.load(f)
ckpt = torch.load(train_summary["checkpoint"], weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
log(f"  Model loaded: epoch {train_summary['best_epoch']}, AUROC={train_summary['auroc']:.4f}, MRR={train_summary['mrr']:.4f}")

# ── Node name lookup ───────────────────────────────────────────────────────
node_names = {}
for nt in data.node_types:
    if hasattr(data[nt], 'name'):
        for idx, name in enumerate(data[nt].name):
            node_names[(nt, idx)] = str(name)

# ── Bidirectional adjacency (for BFS) ──────────────────────────────────────
adj = defaultdict(list)
for et in data.edge_types:
    ei = data[et].edge_index
    for j in range(ei.shape[1]):
        src, dst = int(ei[0,j]), int(ei[1,j])
        adj[(et[0], src)].append((et[2], dst, et[1], 'forward'))
        adj[(et[2], dst)].append((et[0], src, et[1], 'reverse'))

# ── Pure forward pass: extract attention weights ────────────────────────────
log("\n[2/5] Extracting attention weights (forward pass, no grad)...")
x_dict = {nt: data[nt].x for nt in data.node_types}
ei_dict = {et: data[et].edge_index for et in top20}

with torch.no_grad():
    embeddings, layer_attns = model(x_dict, ei_dict, cos_decay=0.0,
                                     return_attention=True)

# Merge attention across layers: average L1 + L2 attention per edge
merged_attn = {}
for et in top20:
    attn_list = []
    for layer_attn in layer_attns:
        if et in layer_attn:
            attn_list.append(layer_attn[et])
    if attn_list:
        merged_attn[et] = sum(attn_list) / len(attn_list)
log(f"  Merged attention for {len(merged_attn)} edge types across {len(layer_attns)} layers")

# Build per-edge attention lookup: (src_type, dst_type, src_idx, dst_idx) -> weight
edge_attn_lookup = {}
for et in top20:
    if et not in merged_attn:
        continue
    ei = data[et].edge_index
    aw = merged_attn[et]  # [E]
    for j in range(ei.shape[1]):
        src, dst = int(ei[0,j]), int(ei[1,j])
        edge_attn_lookup[(et[0], et[2], src, dst, et[1])] = float(aw[j])

log(f"  Edge attention lookup: {len(edge_attn_lookup):,} entries")

# ── Mechanism anchors with cascade step mapping ─────────────────────────────
MECHANISM_ANCHORS = {
    'fut8':                 {'step': 1, 'label': 'Core Fucosylation'},
    'lgals3':               {'step': 2, 'label': 'Galectin-3 / ECM'},
    'galectin-3':           {'step': 2, 'label': 'Galectin-3 / ECM'},
    'cd44':                 {'step': 3, 'label': 'Cell Adhesion'},
    'itgb1':                {'step': 3, 'label': 'Cell Adhesion'},
    'icam1':                {'step': 3, 'label': 'Cell Adhesion'},
    'vcam1':                {'step': 3, 'label': 'Cell Adhesion'},
    'rhoa':                 {'step': 4, 'label': 'Cytoskeletal Signaling'},
    'rock1':                {'step': 4, 'label': 'Cytoskeletal Signaling'},
    'rock2':                {'step': 4, 'label': 'Cytoskeletal Signaling'},
    'mapk1':                {'step': 5, 'label': 'MAPK Signal Transduction'},
    'mapk3':                {'step': 5, 'label': 'MAPK Signal Transduction'},
    'mapk8':                {'step': 5, 'label': 'MAPK Signal Transduction'},
    'mapk14':               {'step': 5, 'label': 'MAPK Signal Transduction'},
    'nfkb1':                {'step': 6, 'label': 'Inflammatory Transcription'},
    'rela':                 {'step': 6, 'label': 'Inflammatory Transcription'},
    'nfkb2':                {'step': 6, 'label': 'Inflammatory Transcription'},
    'stat3':                {'step': 6, 'label': 'Inflammatory Transcription'},
    'fos':                  {'step': 6, 'label': 'Inflammatory Transcription'},
    'jun':                  {'step': 6, 'label': 'Inflammatory Transcription'},
}

# Broad anchor search: also covers mechanism-related processes/diseases
BROAD_ANCHOR_TERMS = [
    'vascular smooth muscle', 'chronic inflammation', 'catheter-related thrombosis',
    'endothelial', 'platelet', 'coagulation', 'fibrin', 'thrombosis', 'fibrosis',
    'inflammat', 'cytokine', 'chemokine', 'adhesion molecule',
    'extracellular matrix', 'ecm', 'collagen', 'tgf',
]

ANCHOR_SEARCH = list(MECHANISM_ANCHORS.keys())

def is_anchor(node_name):
    """Check if a node matches cascade anchor genes or broad mechanism terms."""
    nl = node_name.lower()
    for a in ANCHOR_SEARCH:
        if a in nl:
            return a, 'gene'
    for term in BROAD_ANCHOR_TERMS:
        if term in nl:
            return term, 'term'
    return None, None

# ── Load targets from PCA model ─────────────────────────────────────────────
log("\n[3/5] Loading top-5 hidden targets from PCA model...")
with open("figures/pca_hidden/hidden_top15.json") as f:
    hidden_targets = json.load(f)
targets = hidden_targets[:5]
for i, t in enumerate(targets):
    log(f"  #{i+1}: {t['target']} ({t['type']}) score={t['gnn_score']:.1f} anchor={t['anchor']}")

# ── Find node in KG ─────────────────────────────────────────────────────────
def find_node(search_term, preferred_type=None):
    results = []
    for (nt, idx), name in node_names.items():
        nl = name.lower()
        if search_term in nl or nl in search_term:
            if preferred_type and nt != preferred_type:
                continue
            deg = len(adj.get((nt, idx), []))
            results.append((nt, idx, name, deg))
    if not results:
        # Try without preferred type
        for (nt, idx), name in node_names.items():
            nl = name.lower()
            if search_term in nl or nl in search_term:
                deg = len(adj.get((nt, idx), []))
                results.append((nt, idx, name, deg))
    results.sort(key=lambda x: -x[3])  # highest degree first
    return results[0] if results else None

# ── Attention-guided BFS ────────────────────────────────────────────────────
def attention_guided_bfs(start_node, max_depth=4, top_k_per_hop=5):
    """BFS from start_node, at each hop following top-k highest-attention edges.

    Returns list of paths from start to anchor nodes.
    """
    paths = []
    nt_start, idx_start = start_node
    visited = {(nt_start, idx_start): [[(nt_start, idx_start)]]}
    q = deque([(nt_start, idx_start)])

    while q:
        cur_nt, cur_idx = q.popleft()
        cur_paths = visited[(cur_nt, cur_idx)]
        cur_depth = len(cur_paths[0]) - 1
        if cur_depth >= max_depth:
            continue

        # Collect all neighbors with attention weights
        neighbors = []
        for nbr in adj.get((cur_nt, cur_idx), []):
            nbr_type, nbr_idx, rel, direction = nbr
            nk = (nbr_type, nbr_idx)
            if any(nk in p for p in cur_paths):  # no cycles
                continue

            # Get attention weight
            if direction == 'forward':
                atw = edge_attn_lookup.get((cur_nt, nbr_type, cur_idx, nbr_idx, rel), 0.0)
            else:
                atw = edge_attn_lookup.get((nbr_type, cur_nt, nbr_idx, cur_idx, rel), 0.0)
            neighbors.append((nk, rel, direction, atw))

        if not neighbors:
            continue

        # Sort by attention weight descending, take top-k
        neighbors.sort(key=lambda x: -x[3])
        top_neighbors = neighbors[:top_k_per_hop]

        for nk, rel, direction, atw in top_neighbors:
            nn = node_names.get(nk, '').lower()
            anchor_hit, anchor_type = is_anchor(nn)

            new_paths = [p + [(nk, rel, direction, atw)] for p in cur_paths]

            if nk not in visited:
                visited[nk] = new_paths
                q.append(nk)
            else:
                visited[nk].extend(new_paths)

            if anchor_hit:
                for np in new_paths:
                    anchor_step = MECHANISM_ANCHORS.get(anchor_hit, {}).get('step', 99)
                    anchor_label = MECHANISM_ANCHORS.get(anchor_hit, {}).get('label', anchor_hit.replace('_', ' ').title())
                    paths.append({
                        'path': np,
                        'anchor': anchor_hit,
                        'anchor_type': anchor_type,
                        'anchor_step': anchor_step,
                        'anchor_label': anchor_label,
                        'length': len(np),
                    })

    return paths

# ── Build subgraphs ─────────────────────────────────────────────────────────
log("\n[4/5] Building attention-guided mechanism subgraphs...")
os.makedirs("figures/pca_hidden/subgraphs", exist_ok=True)

all_results = []

for tgt in targets:
    search_term = tgt['target']
    log(f"\n  Target: {tgt['target']} ({tgt['type']})")

    node = find_node(search_term, preferred_type=tgt.get('type'))
    if node is None:
        log(f"    NOT FOUND in KG")
        continue

    nt, idx, full_name, deg = node
    log(f"    Found: [{nt}] {full_name[:80]} (degree={deg})")

    # Attention-guided BFS to anchors
    paths = attention_guided_bfs((nt, idx), max_depth=4, top_k_per_hop=5)
    log(f"    Anchor paths found: {len(paths)}")

    # Group by anchor, select best path per anchor
    anchor_best = defaultdict(list)
    for p in paths:
        anchor_best[p['anchor']].append(p)

    best_paths = []
    for anchor, plist in anchor_best.items():
        # Select: shortest path, tie-break by max attention weight
        best = min(plist, key=lambda x: (x['length'], -sum(step[3] for step in x['path'][1:])))
        best_paths.append(best)
    best_paths.sort(key=lambda x: (x['anchor_step'], x['length']))

    # Build subgraph nodes & edges from best paths (top 8 paths for clarity)
    subgraph_nodes = {}
    subgraph_edges = []

    for bp in best_paths[:8]:
        # First element is start node (2-tuple: (nt, idx)), rest are (nk, rel, dir, atw)
        for step in bp['path']:
            nk = step[0]  # first element is always node key
            if nk not in subgraph_nodes:
                nn = node_names.get(nk, f"{nk[0]}_{nk[1]}")
                a_hit, _ = is_anchor(nn.lower() if nn else '')
                subgraph_nodes[nk] = {
                    'id': nn[:80],
                    'type': nk[0],
                    'is_target': (nk == (nt, idx)),
                    'is_anchor': a_hit is not None,
                }

        # Walk path, add edges (skip first element, start from i=1)
        for i in range(1, len(bp['path'])):
            prev = bp['path'][i - 1]
            cur = bp['path'][i]
            prev_nk = prev[0]
            cur_nk, cur_rel, cur_dir, cur_atw = cur

            prev_nn = node_names.get(prev_nk, str(prev_nk))
            cur_nn = node_names.get(cur_nk, str(cur_nk))

            subgraph_edges.append({
                'source': prev_nn[:80],
                'target': cur_nn[:80],
                'source_type': prev_nk[0],
                'target_type': cur_nk[0],
                'relation': cur_rel if cur_dir == 'forward' else f"INVERSE_{cur_rel}",
                'attention_weight': round(float(cur_atw), 6),
                'hop': i,
                'anchor': bp['anchor'],
                'anchor_type': bp.get('anchor_type', 'gene'),
                'anchor_step': bp['anchor_step'],
                'anchor_label': bp['anchor_label'],
            })

    # Map node cascade steps
    for nk, nd in subgraph_nodes.items():
        a_hit, _ = is_anchor(nd['id'].lower())
        if a_hit and a_hit in MECHANISM_ANCHORS:
            nd['cascade_step'] = MECHANISM_ANCHORS[a_hit]['step']
            nd['cascade_label'] = MECHANISM_ANCHORS[a_hit]['label']

    result = {
        'target': tgt['target'],
        'target_type': nt,
        'full_name': full_name[:80],
        'gnn_score': tgt['gnn_score'],
        'degree': deg,
        'num_anchor_paths': len(paths),
        'shortest_path_length': min(len(p['path']) for p in paths) if paths else None,
        'anchor_cascade_map': [
            {'anchor': bp['anchor'], 'step': bp['anchor_step'],
             'label': bp['anchor_label'], 'hops': bp['length']}
            for bp in best_paths[:8]
        ],
        'nodes': list(subgraph_nodes.values()),
        'edges': subgraph_edges,
    }
    all_results.append(result)

    # Save individual
    fname = tgt['target'].replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    with open(f"figures/pca_hidden/subgraphs/{fname}.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print mechanism mapping summary
    log(f"    Mechanism cascade mapping:")
    for bp in best_paths[:5]:
        path_nodes = " -> ".join([
            f"{node_names.get(step[0], str(step[0]))[:25]}"
            for step in bp['path']
        ])
        log(f"      [{bp['anchor_label']}] (step {bp['anchor_step']}, {bp['length']} hops): {path_nodes}")

# ── Master summary ──────────────────────────────────────────────────────────
log("\n[5/5] Exporting master summary...")
summary = {
    'method': 'native HGT attention weights (no grad, pure forward pass)',
    'model': f"PCA PubMedBERT->128d, epoch {train_summary['best_epoch']}, AUROC={train_summary['auroc']:.4f}",
    'targets': [
        {'rank': i+1, 'name': r['target'], 'degree': r['degree'],
         'paths': r['num_anchor_paths'],
         'shortest_hops': r['shortest_path_length'],
         'cascade_mapping': [m['label'] for m in r['anchor_cascade_map'][:3]],
         'output_file': f"figures/pca_hidden/subgraphs/{r['target'].replace(' ','_').replace('(','').replace(')','').replace('/','_')}.json"}
        for i, r in enumerate(all_results)
    ],
    'mechanism_cascade': [
        {'step': 1, 'label': 'Core Fucosylation', 'anchors': ['FUT8']},
        {'step': 2, 'label': 'Galectin-3 / ECM', 'anchors': ['Lgals3']},
        {'step': 3, 'label': 'Cell Adhesion', 'anchors': ['CD44', 'ITGB1']},
        {'step': 4, 'label': 'Cytoskeletal Signaling', 'anchors': ['RhoA', 'ROCK1', 'ROCK2']},
        {'step': 5, 'label': 'MAPK Signal Transduction', 'anchors': ['MAPK1', 'MAPK3']},
        {'step': 6, 'label': 'Inflammatory Transcription', 'anchors': ['NFKB1', 'RELA']},
    ],
    'total_targets_analyzed': len(all_results),
    'timestamp': datetime.now().isoformat(),
}

with open("figures/pca_hidden/subgraphs/_summary.json", "w", encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

t = time.time() - t0
log(f"\nDone in {t/60:.1f} min — memory-safe, zero backward pass")
log(f"Subgraphs saved: figures/pca_hidden/subgraphs/")
log(f"Mechanism cascade alignment: FUT8(step1)→Lgals3(step2)→CD44(step3)→RhoA/ROCK(step4)→MAPK(step5)→NF-kB(step6)")
