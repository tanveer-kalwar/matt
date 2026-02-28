"""
Automatic dataset download and preprocessing for all 33 benchmarks.

Handles:
    - OpenML download via sklearn or openml package
    - UCI download via ucimlrepo for datasets not on OpenML
    - Target column binarization for binary variants (abalone_19, etc.)
    - Missing value imputation (median for numeric, mode for categorical)
    - Consistent preprocessing: QuantileTransform
    - Stratified train/test split
"""

import os
import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Dict, Optional
from collections import Counter

from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from .config import DATASET_REGISTRY

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _clean_target_values(arr: np.ndarray) -> np.ndarray:
    """Strip whitespace AND trailing punctuation (dots) from target values.

    OpenML id=38 (thyroid sick) encodes targets as 'sick.' and 'negative.'
    with trailing dots. Plain .strip() does not remove them.
    """
    cleaned = []
    for v in arr:
        s = str(v).strip()
        # Strip trailing dots that appear in thyroid and similar datasets
        s = s.rstrip(".")
        s = s.strip()
        cleaned.append(s)
    return np.array(cleaned)


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


def _download_uci(uci_id: int, dataset_name: str) -> pd.DataFrame:
    """Download from UCI Machine Learning Repository via ucimlrepo package.

    Used for datasets not correctly available on OpenML:
        - Dry_Beans (UCI id=602): 13,611 samples, 7 classes
    """
    cache_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        # Validate: if cached file is wrong dataset (e.g., old 589-sample version), re-download
        if dataset_name == "Dry_Beans" and len(df) < 10000:
            print(f"    Cached {dataset_name} has only {len(df)} rows — re-downloading from UCI...")
            os.remove(cache_path)
        else:
            return df

    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=uci_id)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        df.to_csv(cache_path, index=False)
        print(f"    Downloaded {dataset_name} from UCI (id={uci_id})")
        return df
    except ImportError:
        raise RuntimeError(
            f"ucimlrepo package required for {dataset_name}.\n"
            f"Install: pip install ucimlrepo\n"
            f"Then retry."
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot download {dataset_name} from UCI (id={uci_id}): {e}"
        )


def _impute_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical.

    CRITICAL: Use imputation instead of dropna() for datasets with many
    missing values (e.g., thyroid_sick has ~30% missing per feature).
    dropna() on thyroid_sick removes ~95% of rows, leaving an empty
    dataset that causes 'list index out of range' in Counter().most_common().
    """
    result = X_df.copy()

    numeric_cols = result.select_dtypes(include=np.number).columns.tolist()
    cat_cols = result.select_dtypes(exclude=np.number).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        result[numeric_cols] = num_imputer.fit_transform(result[numeric_cols])

    for col in cat_cols:
        mode_val = result[col].mode()
        if len(mode_val) > 0:
            result[col] = result[col].fillna(mode_val.iloc[0])
        else:
            result[col] = result[col].fillna("missing")

    return result


def _apply_minority_rule(y: np.ndarray, rule: str, df: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Convert multiclass targets to binary using the specified rule."""
    # Use _clean_target_values instead of plain str(v).strip() to handle
    # trailing dots in thyroid_sick targets ("sick." → "sick")
    y_str = _clean_target_values(y)

    if rule == "minority":
        counts = Counter(y_str)
        if len(counts) == 0:
            raise ValueError("Empty target array — cannot apply minority rule.")
        majority = counts.most_common(1)[0][0]
        return (y_str != majority).astype(int)

    elif rule.startswith("eq_"):
        val = rule[3:]
        binary = (y_str == val).astype(int)
        # If string match yields no positives, try numeric comparison
        # This handles cases where int 19 becomes "19.0" via str()
        if binary.sum() == 0:
            try:
                num_val = float(val)
                y_num = pd.to_numeric(pd.Series(y_str), errors="coerce")
                binary = (y_num == num_val).fillna(False).astype(int).values
            except (ValueError, TypeError):
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
        parts = rule[5:].split("_")
        mask = np.isin(y_str, parts)
        return mask

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

    # Determine source and load accordingly
    source = info.get("source", "openml")

    # Try local CSV first (respects both openml and uci cached files)
    local_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")

    if source == "uci":
        uci_id = info["uci_id"]
        df = _download_uci(uci_id, dataset_name)
    elif os.path.exists(local_path):
        df = pd.read_csv(local_path)
        # Validate cached file is not stale/wrong
        if dataset_name == "Dry_Beans" and len(df) < 10000:
            print(f"    Stale cache ({len(df)} rows) — re-downloading from UCI...")
            os.remove(local_path)
            df = _download_uci(info.get("uci_id", 602), dataset_name)
        else:
            print(f"    Loaded from {local_path}")
    else:
        openml_id = info["openml_id"]
        df = _download_openml(openml_id, dataset_name)

    # Replace common missing value markers with NaN
    df = df.replace(["?", " ?", "NA", "N/A", "nan", "NaN", ""], np.nan)

    # Drop rows where TARGET is missing (only — features will be imputed)
    target_col = info.get("target", df.columns[-1])
    if target_col not in df.columns:
        target_col = df.columns[-1]

    df = df.dropna(subset=[target_col])

    y_raw = df[target_col].values
    X_df = df.drop(columns=[target_col])

    # Impute missing feature values (instead of dropping rows)
    # CRITICAL for thyroid_sick which has ~30% missing per feature
    X_df = _impute_features(X_df)

    # Apply minority rule
    rule = info.get("minority_rule", "minority")
    is_binary = info.get("binary", True)

    if is_binary and rule.startswith("pair_") and not rule.startswith("pair_AB"):
        mask = _apply_minority_rule(y_raw, rule)
        X_df = X_df[mask].reset_index(drop=True)
        y_raw = y_raw[mask]
        y_str = _clean_target_values(y_raw)
        parts = rule[5:].split("_")
        y = (y_str == parts[-1]).astype(int)
    elif is_binary:
        y = _apply_minority_rule(y_raw, rule)
    else:
        le = LabelEncoder()
        y = le.fit_transform(_clean_target_values(y_raw))

    # Validate that at least 2 classes with sufficient samples exist
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        y = _apply_minority_rule(y_raw, "minority")
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2 or counts.min() < 2:
            raise ValueError(
                f"Dataset {dataset_name}: only {len(unique)} class(es) found after "
                f"binarization — cannot form a valid train/test split."
            )

    # Encode features: numeric as-is, categorical via LabelEncoder
    numeric_cols = X_df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_df.select_dtypes(exclude=np.number).columns.tolist()

    parts_list = []
    if numeric_cols:
        parts_list.append(X_df[numeric_cols].values.astype(float))
    for col in cat_cols:
        le = LabelEncoder()
        parts_list.append(le.fit_transform(X_df[col].astype(str)).reshape(-1, 1).astype(float))

    if parts_list:
        X = np.hstack(parts_list)
    else:
        raise ValueError(f"No features found in {dataset_name}")

    # Stratified split
    unique_y, counts_y = np.unique(y, return_counts=True)
    min_count = counts_y.min()
    n_test = max(1, int(len(y) * 0.2))
    min_samples_per_class_for_stratify = max(2, int(np.ceil(n_test / len(unique_y))))
    can_stratify_split = min_count >= min_samples_per_class_for_stratify
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed,
        stratify=y if (min_count >= 2 and can_stratify_split) else None
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
