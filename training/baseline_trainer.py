# training/baseline_trainer.py
"""Unified FairTrainer for baseline comparison with locked hyperparameters.

Ensures all models are trained with identical config, preventing reviewer
criticism of hyperparameter unfairness. Supports ablation mode switching.
"""
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional
from torch_geometric.data import HeteroData

from .link_prediction import LinkPredictionTrainer


class FairTrainer:
    """Train multiple models under identical hyperparameter constraints.

    Args:
        models: {"Model Name": nn.Module} dict
        unified_config: Hyperparameters LOCKED for all models
        device: "cpu" or "cuda"
        ablation_mode: {"Model Name": "mode_name"} per-model ablation config
        ablation_config_path: Path to ablation_config.yaml
    """

    def __init__(self, models: Dict[str, torch.nn.Module],
                 unified_config: dict, device: str = "cpu",
                 ablation_mode: Optional[Dict[str, str]] = None,
                 ablation_config_path: str = "config/ablation_config.yaml"):
        self.models = models
        self.config = unified_config
        self.device = device
        self.ablation_mode = ablation_mode or {}
        self.ablation_config_path = ablation_config_path

        self.ablation_cfg = {}
        if Path(ablation_config_path).exists():
            with open(ablation_config_path, encoding="utf-8") as f:
                self.ablation_cfg = yaml.safe_load(f)

    def train_all(self, data: HeteroData, train_ei: Dict, val_ei: Dict,
                  neg_ei: Dict, edge_weight_bias: Optional[Dict] = None,
                  verbose: bool = True) -> Dict[str, dict]:
        """Train all models under identical conditions.

        Returns:
            {"Model Name": {"best_epoch": int, "best_val_mrr": float, "history": list}}
        """
        results = {}

        for name, model in self.models.items():
            if verbose:
                print(f"\n{'='*50}")
                print(f"Training: {name}")
                print(f"{'='*50}")

            mode = self.ablation_mode.get(name, "full_prior")
            mode_cfg = self.ablation_cfg.get("ablation_modes", {}).get(mode, {})
            use_cos_decay = mode_cfg.get("cos_decay_enabled", True)
            use_edge_bias = mode_cfg.get("edge_bias_enabled", True)

            model_bias = edge_weight_bias if use_edge_bias else None

            trainer = LinkPredictionTrainer(
                model=model,
                learning_rate=self.config["learning_rate"],
                num_epochs=self.config["num_epochs"],
                patience=self.config["patience"],
                device=self.device,
                checkpoint_dir=f"checkpoints/{name.replace(' ', '_')}",
                edge_weight_bias=model_bias,
                batch_size=self.config.get("batch_size", 256),
                num_neighbors=self.config.get("num_neighbors", [10, 5, 5]),
            )

            # Override cos_decay for ablation modes that disable it
            if not use_cos_decay:
                class ZeroDecay:
                    def __call__(self, step):
                        return 0.0
                trainer.cos_decay = ZeroDecay()

            result = trainer.fit(data, train_ei, val_ei, neg_ei, verbose=verbose)
            results[name] = result

            if verbose:
                print(f"  {name}: best_epoch={result['best_epoch']}, "
                      f"best_val_mrr={result['best_val_mrr']:.4f}")

        return results
