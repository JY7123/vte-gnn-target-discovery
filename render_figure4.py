#!/usr/bin/env python3
"""Figure 4 renderer: attention-guided mechanism subgraphs -> 300 DPI network figures.

Memory-safe: processes one target at a time, closes figures between renders.
Color-coded by FUT8->NF-kB cascade step.
"""
import json, os, time
from datetime import datetime
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

def log(msg):
    print(msg, flush=True)

CASCADE_COLORS = {
    1: '#1a9641',  # FUT8 - dark green
    2: '#66c2a5',  # Lgals3 - light green
    3: '#4575b4',  # CD44/ITGB1 - blue
    4: '#d73027',  # RhoA/ROCK - red
    5: '#f46d43',  # MAPK - orange
    6: '#542788',  # NF-kB/STAT3 - purple
    99: '#bdbdbd', # broad term - grey
}
TARGET_COLOR = '#ffff33'  # yellow
INTERMEDIATE_COLOR = '#f0f0f0'  # light grey

CASCADE_LABELS = {
    1: 'Core Fucosylation\n(FUT8)',
    2: 'Galectin-3/ECM\n(Lgals3)',
    3: 'Cell Adhesion\n(CD44, ITGB1)',
    4: 'Cytoskeletal Signaling\n(RhoA, ROCK1/2)',
    5: 'MAPK Transduction\n(MAPK1/3)',
    6: 'Inflammatory Transcription\n(NFKB1, RELA, STAT3)',
    99: 'Broad Mechanism',
}

def short_name(name, max_len=16):
    """Abbreviate long node names."""
    name = str(name)
    if len(name) <= max_len:
        return name
    # Try to find meaningful abbreviation
    words = name.replace('-', ' ').replace('_', ' ').split()
    if len(words) >= 2:
        abbr = words[0][:6] + '...' + words[-1][:6]
        if len(abbr) <= max_len + 2:
            return abbr
    return name[:max_len-2] + '..'

def render_subgraph(subgraph_json_path, output_path, target_name, dpi=300):
    """Render a single subgraph JSON as a network figure."""
    with open(subgraph_json_path, encoding='utf-8') as f:
        data = json.load(f)

    G = nx.DiGraph()
    node_colors = []
    node_sizes = []
    node_labels = {}
    edge_colors = []
    edge_widths = []
    edge_alphas = []

    # Build node set — ensure node IDs are unique strings
    node_id_map = {}  # maps original id to graph-safe id
    for n in data.get('nodes', []):
        nid = str(n['id'])
        # Deduplicate: append type if name collision
        safe_id = nid
        if safe_id in G:
            safe_id = f"{nid}[{n.get('type', '?')}]"
        node_id_map[n['id']] = safe_id
        G.add_node(safe_id)

        if n.get('is_target'):
            node_colors.append(TARGET_COLOR)
            node_sizes.append(800)
        elif n.get('cascade_step') and n['cascade_step'] != 99:
            step = n['cascade_step']
            node_colors.append(CASCADE_COLORS.get(step, INTERMEDIATE_COLOR))
            node_sizes.append(550)
        elif n.get('is_anchor') or n.get('cascade_step') == 99:
            node_colors.append(CASCADE_COLORS[99])
            node_sizes.append(450)
        else:
            node_colors.append(INTERMEDIATE_COLOR)
            node_sizes.append(300)

        node_labels[node_id_map[n['id']]] = short_name(n['id'])

    # Build edges
    max_atw = max((e.get('attention_weight', 0.01) for e in data.get('edges', [])), default=1.0)
    for e in data.get('edges', []):
        src = str(e['source'])
        dst = str(e['target'])
        # Map to safe IDs
        src_id = node_id_map.get(e['source'], src)
        dst_id = node_id_map.get(e['target'], dst)
        # Skip if node wasn't in nodes list
        if src_id not in G or dst_id not in G:
            continue
        G.add_edge(src_id, dst_id)

        aw = e.get('attention_weight', 0.01)
        # Normalize attention weight for visual scaling
        norm_aw = aw / max_atw if max_atw > 0 else 0.5

        step = e.get('anchor_step', 99)
        edge_colors.append(CASCADE_COLORS.get(step, '#bdbdbd'))
        edge_widths.append(0.5 + norm_aw * 3.5)  # 0.5 to 4.0
        edge_alphas.append(0.3 + norm_aw * 0.7)   # 0.3 to 1.0

    if G.number_of_nodes() == 0:
        log(f"  WARNING: No nodes in {subgraph_json_path}")
        return

    # Layout
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_facecolor('#fafafa')

    # Use hierarchical layout for cascade flow
    # Separate nodes by cascade step for layered layout
    step_nodes = defaultdict(list)
    for n in data.get('nodes', []):
        step = n.get('cascade_step', 0)
        if n.get('is_target'):
            step = 0  # target at top
        safe_id = node_id_map.get(n['id'], str(n['id']))
        if safe_id in G:
            step_nodes[step].append(safe_id)

    # Build layered positions
    pos = {}
    steps_sorted = sorted(step_nodes.keys())
    for layer_idx, step in enumerate(steps_sorted):
        nodes = step_nodes[step]
        y = 1.0 - (layer_idx + 0.5) / len(steps_sorted)
        for i, nid in enumerate(nodes):
            x = (i + 1) / (len(nodes) + 1)
            pos[nid] = (x, y)

    # If only a few steps, use spring layout as fallback
    if len(steps_sorted) <= 1:
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    # Draw
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                          width=edge_widths, alpha=edge_alphas,
                          arrows=True, arrowsize=12, arrowstyle='-|>',
                          connectionstyle='arc3,rad=0.1')

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors='#333333',
                           linewidths=0.8)

    nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels,
                           font_size=7, font_weight='bold',
                           font_family='sans-serif')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=TARGET_COLOR, edgecolor='#333', label='Hidden Target')
    ]
    for step in range(1, 7):
        legend_elements.append(
            mpatches.Patch(facecolor=CASCADE_COLORS[step], edgecolor='#333',
                          label=CASCADE_LABELS[step])
        )

    ax.legend(handles=legend_elements, loc='upper left', fontsize=7,
             framealpha=0.9, ncol=1, bbox_to_anchor=(1.01, 1.0))

    # Title
    mechanism_info = ''
    cascade_map = data.get('anchor_cascade_map', [])
    if cascade_map:
        steps_hit = sorted(set(m['step'] for m in cascade_map if m['step'] != 99))
        mechanism_info = f" | Cascade: {', '.join(f'step{s}' for s in steps_hit)}"

    ax.set_title(f"{target_name}\n{data.get('full_name', '')}{mechanism_info}",
                fontsize=11, fontweight='bold', pad=12)
    ax.axis('off')

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

def main():
    t0 = time.time()
    log("=" * 60)
    log(f"Figure 4 Renderer — 300 DPI   {datetime.now():%H:%M:%S}")
    log("=" * 60)

    subgraph_dir = "figures/pca_hidden/subgraphs"
    output_dir = "figures/pca_hidden/renders"
    os.makedirs(output_dir, exist_ok=True)

    # Load summary
    with open(f"{subgraph_dir}/_summary.json", encoding='utf-8') as f:
        summary = json.load(f)

    targets_data = summary.get('targets', [])
    log(f"Rendering {len(targets_data)} subgraphs...")

    for i, tgt in enumerate(targets_data):
        name = tgt['name']
        json_path = tgt['output_file']
        png_path = f"{output_dir}/{name.replace(' ','_').replace('-','_')}.png"

        log(f"  [{i+1}/{len(targets_data)}] {name}...")
        render_subgraph(json_path, png_path, name, dpi=300)
        log(f"    -> {png_path}")

    # Render combined cascade overview
    log(f"\n  Rendering cascade overview...")
    render_cascade_overview(output_dir, targets_data)

    t = time.time() - t0
    log(f"\nDone in {t:.1f}s — {len(targets_data)} figures at 300 DPI")
    log(f"Figures: {output_dir}/")

def render_cascade_overview(output_dir, targets_data):
    """Render a summary figure showing all targets mapped to cascade steps."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.set_facecolor('#fafafa')

    # Draw cascade steps as horizontal bars
    y_positions = list(range(6, 0, -1))
    bar_height = 0.6

    for step, y in enumerate(y_positions, 1):
        ax.barh(y, 1.0, height=bar_height, color=CASCADE_COLORS[step],
               edgecolor='#333', linewidth=1.5, alpha=0.7)
        ax.text(-0.02, y, CASCADE_LABELS[step].replace('\n', ' '),
               ha='right', va='center', fontsize=9, fontweight='bold')

    # Map targets to cascade steps
    for i, tgt in enumerate(targets_data):
        name = tgt['name']
        cascade_mapping = tgt.get('cascade_mapping', [])
        paths = tgt.get('paths', 0)

        for label in cascade_mapping:
            # Find which step this label corresponds to
            for step in range(1, 7):
                if CASCADE_LABELS[step].startswith(label):
                    y = y_positions[step - 1]
                    x = 0.15 + i * 0.17
                    ax.plot(x, y, 'o', color=TARGET_COLOR, markersize=14,
                           markeredgecolor='#333', markeredgewidth=1.5,
                           zorder=10)
                    ax.annotate(name, (x, y + 0.5), fontsize=7,
                              ha='center', va='bottom', rotation=30,
                              fontweight='bold')
                    break

    ax.set_xlim(-0.5, 1.2)
    ax.set_ylim(0.3, 6.7)
    ax.set_title('Hidden Target → FUT8→NF-kB Cascade Mapping\n(Attention-Weighted Mechanism Subgraphs)',
                fontsize=13, fontweight='bold')
    ax.axis('off')

    fig.tight_layout()
    fig.savefig(f"{output_dir}/_cascade_overview.png", dpi=300,
               facecolor='white', edgecolor='none')
    plt.close(fig)
    log(f"    -> {output_dir}/_cascade_overview.png")

if __name__ == "__main__":
    main()
