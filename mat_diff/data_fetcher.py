"""
Automatic dataset download and preprocessing for all 33 benchmarks.

Handles:
    - OpenML download via sklearn or openml package
    - Target column binarization for binary variants (abalone_19, etc.)
    - Consistent preprocessing: imputation, encoding, QuantileTransform
    - Stratified train/test split
"""

import os
import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Dict, Optional
from collections import Counter

from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split

from .config import DATASET_REGISTRY

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _download_openml(openml_id: int, dataset_name: str) -> pd.DataFrame:
    """Download from OpenML, cache locally."""
    cache_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
        df = data.frame
        if df is None:
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
        df.to_csv(cache_path, index=False)
        print(f"    Downloaded {dataset_name} from OpenML (id={openml_id})")
        return df
    except Exception as e:
        raise RuntimeError(
            f"Cannot download {dataset_name} (OpenML id={openml_id}): {e}\n"
            f"Install: pip install scikit-learn openml"
        )


def _apply_minority_rule(y: np.ndarray, rule: str, df: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Convert multiclass targets to binary using the specified rule."""
    y_str = np.array([str(v).strip() for v in y])

    if rule == "minority":
        # Most common = 0, everything else = 1
        counts = Counter(y_str)
        majority = counts.most_common(1)[0][0]
        return (y_str != majority).astype(int)

    elif rule.startswith("eq_"):
        val = rule[3:]
        binary = (y_str == val).astype(int)
        # If string match yields no positives, try numeric comparison
        if binary.sum() == 0:
            try:
                num_val = float(val)
                y_num = pd.to_numeric(pd.Series(y_str), errors="coerce")
                binary = (y_num == num_val).fillna(False).astype(int).values
            except ValueError:
                pass
        return binary

    elif rule.startswith("le_"):
        threshold = int(rule[3:])
        y_num = pd.to_numeric(pd.Series(y_str), errors="coerce").fillna(threshold + 1).values
        return (y_num <= threshold).astype(int)

    elif rule.startswith("ge_"):
        threshold = int(rule[3:])
        y_num = pd.to_numeric(pd.Series(y_str), errors="coerce").fillna(threshold - 1).values
        return (y_num >= threshold).astype(int)

    elif rule.startswith("pair_"):
        # e.g., pair_0_5 → keep only classes 0 and 5
        parts = rule[5:].split("_")
        mask = np.isin(y_str, parts)
        # Return the full mask and labels for filtering
        return mask  # Caller must filter X too

    elif rule == "pair_AB":
        mask = np.isin(y_str, ["A", "B", "1", "2"])
        return mask

    else:
        return LabelEncoder().fit_transform(y_str)


def load_dataset(dataset_name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, preprocess, and split a dataset.

    Returns:
        (X_train, y_train, X_test, y_test) — all numpy arrays, features normalized to [0,1]
    """
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")

    info = DATASET_REGISTRY[dataset_name]
    print(f"  Loading {dataset_name}...")

    # Try local CSV first
    local_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        print(f"    Loaded from {local_path}")
    else:
        df = _download_openml(info["openml_id"], dataset_name)

    # Clean
    df = df.replace(["?", " ?", "NA", ""], np.nan)
    df = df.dropna()

    # Identify target
    target_col = info.get("target", df.columns[-1])
    if target_col not in df.columns:
        target_col = df.columns[-1]

    y_raw = df[target_col].values
    X_df = df.drop(columns=[target_col])

    # Apply minority rule
    rule = info.get("minority_rule", "minority")
    is_binary = info.get("binary", True)

    if is_binary and rule.startswith("pair_"):
        mask = _apply_minority_rule(y_raw, rule)
        X_df = X_df[mask].reset_index(drop=True)
        y_raw = y_raw[mask]
        y_str = np.array([str(v).strip() for v in y_raw])
        parts = rule[5:].split("_")
        y = (y_str == parts[-1]).astype(int)  # Last part is minority
    elif is_binary:
        y = _apply_minority_rule(y_raw, rule)
    else:
        le = LabelEncoder()
        y = le.fit_transform(np.array([str(v).strip() for v in y_raw]))

    # Validate that at least 2 classes with sufficient samples exist
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        # Fall back to majority vs. rest binarization
        y = _apply_minority_rule(y_raw, "minority")
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2 or counts.min() < 2:
            raise ValueError(
                f"Dataset {dataset_name}: only {len(unique)} class(es) found after "
                f"binarization — cannot form a valid train/test split."
            )

    # Encode features
    numeric_cols = X_df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_df.select_dtypes(exclude=np.number).columns.tolist()

    parts = []
    if numeric_cols:
        parts.append(X_df[numeric_cols].values.astype(float))
    for col in cat_cols:
        le = LabelEncoder()
        parts.append(le.fit_transform(X_df[col].astype(str)).reshape(-1, 1).astype(float))

    if parts:
        X = np.hstack(parts)
    else:
        raise ValueError(f"No features found in {dataset_name}")

    # Stratified split — fall back to random split if a class is too small
    unique_y, counts_y = np.unique(y, return_counts=True)
    min_count = counts_y.min()
    n_test = max(1, int(len(y) * 0.2))
    min_samples_per_class_for_stratify = max(2, int(np.ceil(n_test / len(unique_y))))
    has_minimum_samples = min_count >= 2
    can_stratify_split = min_count >= min_samples_per_class_for_stratify
    use_stratify = has_minimum_samples and can_stratify_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if use_stratify else None
    )

    # QuantileTransformer: maps to [0,1] — robust to outliers
    scaler = QuantileTransformer(output_distribution="uniform", random_state=seed)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # Report
    dist = dict(zip(*np.unique(y_tr, return_counts=True)))
    ir = max(dist.values()) / max(1, min(dist.values()))
    print(f"    Samples: {len(X_tr)} train, {len(X_te)} test")
    print(f"    Features: {X_tr.shape[1]}, Classes: {len(dist)}, IR: {ir:.1f}")
    print(f"    Distribution: {dist}")

    return X_tr, y_tr, X_te, y_te
