#!/usr/bin/env python3
"""
MAT-Diff training script.

Usage:
    python -m mat_diff.train --dataset wine --device cuda --epochs 500
    python -m mat_diff.train --dataset all --device cuda
"""
import sys
import os
import argparse
import numpy as np
import torch
import warnings

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mat_diff.config import get_matdiff_config, MATDIFF_DATASET_CONFIGS
from mat_diff.manifold_diffusion import MATDiffPipeline

warnings.filterwarnings("ignore")


def load_dataset_auto(name: str):
    """Load dataset using the existing TabDDPM_Aug loader or fallback to UCI."""
    try:
        from TabDDPM_Aug.data_loader import load_dataset, prepare_data
        df, n_classes = load_dataset(name)
        data = prepare_data(df, seed=42)
        return data["X_train_norm"], data["y_train"], data["X_test_norm"], data["y_test"]
    except Exception:
        pass

    # Fallback: load from UCI directly
    try:
        from ucimlrepo import fetch_ucirepo
        from mat_diff.config import DATASET_FILES
        from sklearn.preprocessing import LabelEncoder, QuantileTransformer
        from sklearn.model_selection import train_test_split
        import pandas as pd

        info = DATASET_FILES.get(name, {})
        if "uci_id" in info:
            ds = fetch_ucirepo(id=info["uci_id"])
            df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        elif "openml_id" in info:
            import openml
            ds = openml.datasets.get_dataset(info["openml_id"])
            X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
            df = pd.concat([X, y], axis=1)
        else:
            raise FileNotFoundError(f"No source configured for {name}")

        df = df.replace(["?", " ?"], np.nan).dropna()
        target_col = info.get("target_col", df.columns[-1])
        if target_col not in df.columns:
            target_col = df.columns[-1]

        le = LabelEncoder()
        y_all = le.fit_transform(df[target_col].astype(str).str.strip())
        X_all = df.drop(columns=[target_col])

        # Numeric only for simplicity
        X_num = X_all.select_dtypes(include=np.number)
        if X_num.shape[1] == 0:
            # All categorical: encode
            for col in X_all.columns:
                X_all[col] = LabelEncoder().fit_transform(X_all[col].astype(str))
            X_num = X_all

        scaler = QuantileTransformer(output_distribution="uniform", random_state=42)
        X_scaled = scaler.fit_transform(X_num.values)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        return X_tr, y_tr, X_te, y_te

    except Exception as e:
        raise RuntimeError(f"Cannot load dataset '{name}': {e}")


def train_single(dataset_name: str, device: str, epochs_override: int = None):
    """Train MAT-Diff on a single dataset."""
    print(f"\n{'=' * 70}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'=' * 70}")

    cfg = get_matdiff_config(dataset_name)
    epochs = epochs_override if epochs_override else cfg["epochs"]

    X_tr, y_tr, X_te, y_te = load_dataset_auto(dataset_name)

    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"  Classes: {len(np.unique(y_tr))}, IR: {cfg.get('ir', '?')}")

    pipeline = MATDiffPipeline(
        device=device,
        d_model=cfg["d_model"],
        d_hidden=cfg.get("d_hidden", cfg["d_model"] * 2),
        n_blocks=cfg["n_blocks"],
        n_heads=cfg["n_heads"],
        n_phases=cfg.get("n_phases", 3),
        total_timesteps=cfg.get("total_timesteps", 1000),
        dropout=cfg.get("dropout", 0.1),
        lr=cfg["lr"],
        privacy_quantile=cfg.get("privacy_quantile", 0.05),
    )

    pipeline.fit(X_tr, y_tr, epochs=epochs, batch_size=cfg["batch_size"])

    # Sample and quick-evaluate
    print("\n  Sampling synthetic data...")
    X_syn, y_syn = pipeline.sample()
    print(f"  Generated: {X_syn.shape}")

    # Quick F1 check
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    X_aug = np.vstack([X_tr, X_syn])
    y_aug = np.hstack([y_tr, y_syn])

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_aug, y_aug)
    y_pred = rf.predict(X_te)
    f1 = f1_score(y_te, y_pred, average="macro")
    print(f"  Quick F1-Macro (RF, augmented): {f1:.4f}")

    rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_orig.fit(X_tr, y_tr)
    y_pred_orig = rf_orig.predict(X_te)
    f1_orig = f1_score(y_te, y_pred_orig, average="macro")
    print(f"  Quick F1-Macro (RF, original):  {f1_orig:.4f}")
    print(f"  Improvement: {f1 - f1_orig:+.4f}")

    # Save
    os.makedirs("results/matdiff_checkpoints", exist_ok=True)
    pipeline.save(f"results/matdiff_checkpoints/{dataset_name}.pt")

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="MAT-Diff Training")
    parser.add_argument("--dataset", type=str, default="wine",
                        help="Dataset name or 'all' for all 18")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 70)
    print("MAT-Diff: Manifold-Aligned Tabular Diffusion — Training")
    print(f"Device: {device}")
    print("=" * 70)

    if args.dataset == "all":
        datasets = list(MATDIFF_DATASET_CONFIGS.keys())
    else:
        datasets = [args.dataset]

    for ds in datasets:
        try:
            train_single(ds, device, args.epochs)
        except Exception as e:
            print(f"\n  FAILED on {ds}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
