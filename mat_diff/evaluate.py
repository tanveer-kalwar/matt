"""
MAT-Diff evaluation pipeline.

Protocol (matches DGOT / your baseline_evaluation_Version3.py):
    - Stratified 5-fold cross-validation
    - 10 random seeds per method
    - 5 classifiers: XGBoost, DecisionTree, LogisticRegression, RandomForest, KNN
    - Metrics: Macro-F1, Accuracy, MCC, BalancedAccuracy, AUC-PR, MMD, KS, DCR, MIA
"""

import numpy as np
import warnings
from collections import Counter
from typing import Dict, List, Optional

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score, average_precision_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist

from .config import CLASSIFIER_PARAMS

warnings.filterwarnings("ignore")


# ── Classifiers from DGOT Table III ──
def build_classifiers(seed: int = 42) -> list:
    """Build the 5 classifiers per DGOT Table III."""
    return [
        ("XGB",   XGBClassifier(random_state=seed,
                                **CLASSIFIER_PARAMS["XGBoost"])),
        ("DTC",   DecisionTreeClassifier(random_state=seed,
                                         **CLASSIFIER_PARAMS["DecisionTree"])),
        ("Logit", LogisticRegression(random_state=seed,
                                     **CLASSIFIER_PARAMS["LogisticRegression"])),
        ("RFC",   RandomForestClassifier(random_state=seed,
                                         **CLASSIFIER_PARAMS["RandomForest"])),
        ("KNN",   KNeighborsClassifier(**CLASSIFIER_PARAMS["KNN"])),
    ]


# ── Utility Metrics ──
def evaluate_utility(X_tr, y_tr, X_te, y_te, seed=42) -> Dict[str, Dict[str, float]]:
    """Evaluate all 5 classifiers. Returns {clf_name: {metric: value}}."""
    results = {}
    for name, clf in build_classifiers(seed):
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        m = {
            "Acc":    accuracy_score(y_te, y_pred),
            "F1":     f1_score(y_te, y_pred, average="macro"),
            "MCC":    matthews_corrcoef(y_te, y_pred),
            "BalAcc": balanced_accuracy_score(y_te, y_pred),
        }
        try:
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_te)
                classes = np.unique(y_te)
                if len(classes) == 2:
                    m["AUC_PR"] = average_precision_score(y_te, y_prob[:, 1])
                else:
                    y_bin = label_binarize(y_te, classes=classes)
                    m["AUC_PR"] = average_precision_score(
                        y_bin, y_prob, average="macro"
                    )
            else:
                m["AUC_PR"] = np.nan
        except Exception:
            m["AUC_PR"] = np.nan
        results[name] = m
    return results


# ── Fidelity Metrics ──
def compute_mmd_rbf(X_real, X_syn, gamma=None):
    if X_syn is None or len(X_syn) == 0:
        return np.nan
    if gamma is None:
        gamma = 1.0 / max(1, X_real.shape[1])
    XX = np.exp(-gamma * cdist(X_real, X_real, "sqeuclidean"))
    YY = np.exp(-gamma * cdist(X_syn, X_syn, "sqeuclidean"))
    XY = np.exp(-gamma * cdist(X_real, X_syn, "sqeuclidean"))
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def compute_ks(X_real, X_syn):
    if X_syn is None or len(X_syn) == 0:
        return np.nan
    ks_vals = []
    for j in range(X_real.shape[1]):
        stat, _ = ks_2samp(X_real[:, j], X_syn[:, j])
        ks_vals.append(stat)
    return float(np.mean(ks_vals))


# ── Privacy Metrics ──
def compute_dcr(X_real, X_syn):
    if X_syn is None or len(X_syn) == 0:
        return np.nan
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(X_real)
    dists, _ = nn.kneighbors(X_syn)
    return float(np.median(dists))


def compute_mia(X_train, X_syn, X_test):
    if X_syn is None or len(X_syn) == 0:
        return np.nan
    try:
        n = min(len(X_train), len(X_test), 500)
        idx_tr = np.random.choice(len(X_train), n, replace=False)
        idx_te = np.random.choice(len(X_test), n, replace=False)
        nn = NearestNeighbors(n_neighbors=5, algorithm="auto")
        nn.fit(X_syn)
        dists_tr, _ = nn.kneighbors(X_train[idx_tr])
        dists_te, _ = nn.kneighbors(X_test[idx_te])
        feats = np.vstack([dists_tr, dists_te])
        y_mia = np.array([1] * n + [0] * n)
        mia_clf = LogisticRegression(max_iter=200)
        scores = cross_val_score(mia_clf, feats, y_mia, cv=3, scoring="accuracy")
        return float(np.mean(scores))
    except Exception:
        return np.nan


# ── Main Evaluation ──
def evaluate_matdiff(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    pipeline,
    n_seeds: int = 10,
    n_folds: int = 5,
    method_name: str = "MAT-Diff",
    verbose: bool = True,
) -> dict:
    """Full evaluation of MAT-Diff following DGOT protocol.

    Args:
        X_raw, y_raw: Full dataset.
        pipeline: A MATDiffPipeline instance (untrained; will be re-trained per fold).
        n_seeds: Number of random seeds.
        n_folds: Number of CV folds.
        method_name: Label for printing.
        verbose: Print progress.

    Returns:
        dict with keys: 'utility', 'fidelity', 'privacy', each containing
        per-classifier and per-metric lists of values.
    """
    clf_names = ["XGB", "DTC", "Logit", "RFC", "KNN"]
    utility_metrics = ["Acc", "F1", "MCC", "BalAcc", "AUC_PR"]
    fidelity_metrics = ["MMD", "KS"]
    privacy_metrics = ["DCR", "MIA"]

    results = {
        "utility": {cn: {m: [] for m in utility_metrics} for cn in clf_names},
        "fidelity": {m: [] for m in fidelity_metrics},
        "privacy": {m: [] for m in privacy_metrics},
    }

    total = n_seeds * n_folds
    done = 0

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"Evaluating {method_name}: {n_seeds} seeds x {n_folds} folds")
        print(f"{'=' * 72}")

    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X_raw, y_raw)):
            X_tr, X_te = X_raw[tr_idx], X_raw[te_idx]
            y_tr, y_te = y_raw[tr_idx], y_raw[te_idx]

            # Re-create and train pipeline for this fold
            from .manifold_diffusion import MATDiffPipeline

            fold_pipeline = MATDiffPipeline(
                device=pipeline.device,
                d_model=pipeline.d_model,
                d_hidden=pipeline.d_hidden,
                n_blocks=pipeline.n_blocks,
                n_heads=pipeline.n_heads,
                n_phases=pipeline.n_phases,
                total_timesteps=pipeline.total_timesteps,
                dropout=pipeline.dropout,
                lr=pipeline.lr,
                weight_decay=pipeline.weight_decay,
                privacy_quantile=pipeline.privacy_quantile,
            )

            try:
                from .config import get_matdiff_config, DATASET_REGISTRY
                fold_epochs = max(300, pipeline._fit_epochs if hasattr(pipeline, '_fit_epochs') else 500)
                fold_batch = pipeline._fit_batch if hasattr(pipeline, '_fit_batch') else 128
                fit_epochs = getattr(pipeline, '_fit_epochs', 500)
                fit_batch = getattr(pipeline, '_fit_batch', 128)
                fold_pipeline.fit(X_tr, y_tr, epochs=fit_epochs, batch_size=fit_batch, verbose=False)
                X_syn, y_syn = fold_pipeline.sample()
            except Exception as e:
                if verbose:
                    print(f"  Seed {seed} Fold {fold}: FAILED - {e}")
                continue

            if len(X_syn) == 0:
                continue

            # Augmented training set
            X_aug = np.vstack([X_tr, X_syn])
            y_aug = np.hstack([y_tr, y_syn])

            # Utility
            clf_res = evaluate_utility(X_aug, y_aug, X_te, y_te, seed=seed)
            for cn in clf_names:
                for m in utility_metrics:
                    results["utility"][cn][m].append(clf_res[cn].get(m, np.nan))

            # Fidelity (compare synthetic to real minority)
            minority_mask = np.isin(y_tr, [
                k for k, v in Counter(y_tr).items()
                if v < max(Counter(y_tr).values())
            ])
            X_minority = X_tr[minority_mask]
            if len(X_minority) > 0:
                results["fidelity"]["MMD"].append(compute_mmd_rbf(X_minority, X_syn))
                results["fidelity"]["KS"].append(compute_ks(X_minority, X_syn))

            # Privacy
            results["privacy"]["DCR"].append(compute_dcr(X_tr, X_syn))
            results["privacy"]["MIA"].append(compute_mia(X_tr, X_syn, X_te))

            done += 1
            if verbose:
                print(f"  Progress: {done}/{total} (Seed {seed}, Fold {fold})", end="\r")

        if verbose:
            print(f"  Seed {seed:>2d} complete ({seed + 1}/{n_seeds})" + " " * 30)

    if verbose:
        _print_results(results, clf_names, utility_metrics, method_name)

    return results


def _fmt(vals):
    v = [x for x in vals if not np.isnan(x)]
    if not v:
        return "--"
    return f"{np.mean(v):.4f} +/- {np.std(v):.4f}"


def _print_results(results, clf_names, utility_metrics, method_name):
    """Print formatted results tables."""
    print(f"\n{'=' * 96}")
    print(f"  {method_name} — Utility (mean +/- std)")
    print(f"{'=' * 96}")
    for met in utility_metrics:
        hdr = f"{'Metric':<8}"
        for cn in clf_names:
            hdr += f" | {cn:>15}"
        hdr += f" | {'AVG':>15}"
        print(hdr)
        print("-" * 96)
        row = f"{met:<8}"
        means = []
        for cn in clf_names:
            vals = results["utility"][cn][met]
            row += f" | {_fmt(vals):>15}"
            clean = [v for v in vals if not np.isnan(v)]
            if clean:
                means.append(np.mean(clean))
        if means:
            row += f" | {np.mean(means):>15.4f}"
        else:
            row += f" | {'--':>15}"
        print(row)
        print()

    print(f"\n  {method_name} — Fidelity & Privacy")
    print("-" * 72)
    print(f"  MMD : {_fmt(results['fidelity']['MMD'])}")
    print(f"  KS  : {_fmt(results['fidelity']['KS'])}")
    print(f"  DCR : {_fmt(results['privacy']['DCR'])}")
    print(f"  MIA : {_fmt(results['privacy']['MIA'])}")
    print("=" * 72)


