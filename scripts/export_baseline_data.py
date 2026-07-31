"""Export train/test triples + entity mapping for baseline KG embedding models."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, json, csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def main():
    data = torch.load(BASE / "data/processed/heterodata.pt", weights_only=False)
    import yaml
    with open(BASE / "config/anchor_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    config_ets = [tuple(et) for et in cfg.get("edge_types", [])]
    valid_nts = [nt for nt in data.node_types if data[nt].num_nodes > 0]
    meta_relations = [et for et in config_ets if et in data.edge_types
                      and et[0] in valid_nts and et[2] in valid_nts]

    from data.temporal_split import RandomStratifiedSplitter
    splitter = RandomStratifiedSplitter(seed=42, edge_types=meta_relations)
    train_ei, val_ei, test_ei = splitter.split(data)
    train_ei = {et: ei for et, ei in train_ei.items() if et in meta_relations}
    test_ei = {et: ei for et, ei in test_ei.items() if et in meta_relations}

    entity_map = {}
    global_id = 0
    for nt in valid_nts:
        names = data[nt].name if hasattr(data[nt], 'name') else []
        for local_idx in range(data[nt].num_nodes):
            name = str(names[local_idx]) if local_idx < len(names) else f"{nt}_{local_idx}"
            entity_map[(nt, local_idx)] = (global_id, name, nt)
            global_id += 1

    out_dir = BASE / "data" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Entity map JSON
    export_map = {}
    for (nt, idx), (gid, name, etype) in entity_map.items():
        export_map[f"{nt}_{idx}"] = {"global_id": gid, "name": name, "type": etype}
    with open(out_dir / "entity_map.json", "w", encoding="utf-8") as f:
        json.dump(export_map, f, indent=2, ensure_ascii=False)

    # Relation map
    rel_list = sorted(set(train_ei.keys()) | set(test_ei.keys()))
    rel_map = {et: i for i, et in enumerate(rel_list)}
    with open(out_dir / "relation_map.json", "w", encoding="utf-8") as f:
        json.dump({f"{s}__{r}__{d}": rid for (s, r, d), rid in rel_map.items()}, f, indent=2)

    # Export triples
    def export_triples(ei_dict, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["head", "relation", "tail", "head_name", "tail_name", "relation_name"])
            for et, ei in ei_dict.items():
                if et not in rel_map: continue
                src_t, rel, dst_t = et
                for j in range(ei.shape[1]):
                    h = entity_map.get((src_t, int(ei[0, j])))
                    t = entity_map.get((dst_t, int(ei[1, j])))
                    if h and t:
                        w.writerow([h[0], rel_map[et], t[0], h[1], t[1], rel])

    export_triples(train_ei, out_dir / "train_triples.csv")
    export_triples(test_ei, out_dir / "test_triples.csv")

    n_train = sum(ei.shape[1] for ei in train_ei.values())
    n_test = sum(ei.shape[1] for ei in test_ei.values())
    print(f"Entities: {len(entity_map)}, Relations: {len(rel_map)}")
    print(f"Train triples: {n_train:,}, Test triples: {n_test:,}")

if __name__ == "__main__":
    main()
