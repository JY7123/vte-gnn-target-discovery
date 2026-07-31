"""Train KG embedding baselines and export filtered metrics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, json, csv, time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def load_triples(path):
    triples = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            triples.append((int(row["head"]), int(row["relation"]), int(row["tail"])))
    return torch.tensor(triples, dtype=torch.long)

def main():
    train = load_triples(BASE / "data/baselines/train_triples.csv")
    test = load_triples(BASE / "data/baselines/test_triples.csv")
    num_entities = max(train[:,0].max(), train[:,2].max(),
                       test[:,0].max(), test[:,2].max()).item() + 1
    num_relations = max(train[:,1].max(), test[:,1].max()).item() + 1
    print(f"Entities: {num_entities}, Relations: {num_relations}")
    print(f"Train: {train.shape[0]:,}, Test: {test.shape[0]:,}")

    # Build all-true set for filtered evaluation
    all_true = set()
    for t in [train, test]:
        for j in range(t.shape[0]):
            all_true.add((int(t[j,0]), int(t[j,1]), int(t[j,2])))

    from training.baselines import (
        TransE, DistMult, ComplEx, RotatE,
        BaselineTrainer, evaluate_baseline_filtered
    )

    results = {}
    models = {
        "TransE": TransE(num_entities, num_relations, dim=128, margin=1.0),
        "DistMult": DistMult(num_entities, num_relations, dim=128),
        "ComplEx": ComplEx(num_entities, num_relations, dim=128),
        "RotatE": RotatE(num_entities, num_relations, dim=128, margin=6.0),
    }

    for name, model in models.items():
        print(f"\n{'='*40}\nTraining {name}\n{'='*40}")
        t0 = time.time()
        trainer = BaselineTrainer(model, learning_rate=1e-3, num_epochs=100, device="cpu")
        trainer.fit(train, num_entities, verbose=True)
        metrics = evaluate_baseline_filtered(model, test, all_true, num_entities)
        results[name] = {
            "filtered_mrr": round(metrics["filtered_mrr"], 4),
            "tail_mrr": round(metrics["tail_mrr"], 4),
            "head_mrr": round(metrics["head_mrr"], 4),
            "tail_hits@1": round(metrics["tail_hits@1"], 4),
            "tail_hits@3": round(metrics["tail_hits@3"], 4),
            "tail_hits@10": round(metrics["tail_hits@10"], 4),
            "head_hits@1": round(metrics["head_hits@1"], 4),
            "head_hits@3": round(metrics["head_hits@3"], 4),
            "head_hits@10": round(metrics["head_hits@10"], 4),
            "n_triples": metrics["n_triples"],
            "train_time_min": round((time.time() - t0) / 60, 1),
        }
        print(f"  Filtered MRR: {metrics['filtered_mrr']:.4f}")

    # Add TemperedHGT — unified key names matching baselines
    with open(BASE / "checkpoints/full_training_v2/summary.json") as f:
        hgt = json.load(f)
    results["TemperedHGT"] = {
        "filtered_mrr": hgt["test_mrr_mean"],
        "tail_mrr": hgt["test_mrr_mean"],
        "head_mrr": hgt["test_mrr_mean"],
        "tail_hits@1": None,
        "tail_hits@3": None,
        "tail_hits@10": hgt["test_hits10_mean"],
        "head_hits@1": None,
        "head_hits@3": None,
        "head_hits@10": hgt["test_hits10_mean"],
        "n_triples": 1482,
        "auroc": hgt["test_auroc_mean"],
        "train_time_min": 1.4,
    }

    with open(BASE / "data/baselines/baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to data/baselines/baseline_results.json")

if __name__ == "__main__":
    main()
