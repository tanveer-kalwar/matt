#!/usr/bin/env python3
"""
GOIO Training - Properly integrated with GOIO repository structure
"""
import os
import re
import sys
import subprocess
import numpy as np
import pandas as pd
import argparse
import json


def setup_goio_repo():
    """Clone and setup GOIO"""
    repo_dir = "/content/GOIO"
    
    if not os.path.isdir(repo_dir):
        print("  Cloning GOIO...")
        subprocess.run(
            "git clone https://github.com/MLneuers/GOIO.git /content/GOIO",
            shell=True, capture_output=True, text=True
        )
    
    print("  Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
         "torch", "torchvision", "einops", "scikit-learn", "scipy",
         "pandas", "numpy", "pyyaml", "tqdm", "rtdl"],
        capture_output=True
    )
    
    # Fix verbose=True in ReduceLROnPlateau across ALL Python files
    print("  Patching verbose parameter...")

    for root, dirs, files in os.walk(repo_dir):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Remove verbose=True from ReduceLROnPlateau
                    modified = re.sub(
                        r'ReduceLROnPlateau\(([^)]*),\s*verbose\s*=\s*True',
                        r'ReduceLROnPlateau(\1',
                        content
                    )
                    
                    # Also handle case where verbose=True is the only or first arg after optimizer
                    modified = re.sub(
                        r'ReduceLROnPlateau\(([^)]*)\s*verbose\s*=\s*True,?\s*',
                        r'ReduceLROnPlateau(\1',
                        modified
                    )
                    
                    if modified != content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(modified)
                except (IOError, OSError):
                    pass

    # Fix column classification in dataprocessing.py:
    #
    # Root cause: data_description() classifies the target column as "ordinal"
    # when counts_len >= 10 (e.g. students_dropout has 31 classes).  data_GOIO()
    # then appends the target column index (== n_features) into num_col_idx
    # because it matches the `ds['type'] == 'ordinal'` branch.  The subsequent
    #   X_num_test = xtest[:, num_col_idx]
    # fails with IndexError because xtest is already stripped of the target
    # column (shape (m, n_features)), so index n_features is out of bounds.
    #
    # Fix strategy (robust):
    #   1. Inject a post-loop block inside data_description() that re-classifies
    #      the last json_SOS entry as "categorical"/"label" regardless of how
    #      many unique values the target column has.  We detect the injection
    #      point using a regex so the fix is insensitive to exact indentation
    #      or minor code-style differences.
    #   2. Belt-and-suspenders: in data_GOIO() limit the columns loop to
    #      [:-1] so the target entry is never added to num_col_idx / cat_col_idx
    #      even if the JSON is somehow wrong.

    dp_path = os.path.join(repo_dir, "data", "dataprocessing.py")
    if os.path.exists(dp_path):
        try:
            with open(dp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            modified = content

            # ----------------------------------------------------------------
            # Fix 1 – inject post-loop label-fix into data_description().
            # We insert a block just before the `json_SOS_final = {...}` line.
            # The injected block ensures json_SOS[-1] is always
            # {"name": "label", "type": "categorical", ...} regardless of the
            # number of unique class values.
            # ----------------------------------------------------------------
            _LABEL_FIX_MARKER = "# _goio_label_fix_applied"
            if _LABEL_FIX_MARKER not in modified:
                # Build the injection text using the same indentation as the
                # json_SOS_final line (captured via regex group 1).
                _inject = (
                    "{indent}" + _LABEL_FIX_MARKER + "\n"
                    "{indent}# Ensure last column (target) is always label/categorical\n"
                    "{indent}if json_SOS and (\n"
                    "{indent}        json_SOS[-1].get('type') != 'categorical' or\n"
                    "{indent}        json_SOS[-1].get('name') != 'label'):\n"
                    "{indent}    _last_vals = sorted(list(Counter(data[:, -1]).keys()))\n"
                    "{indent}    json_SOS[-1] = {\n"
                    "{indent}        'i2s': _last_vals,\n"
                    "{indent}        'name': 'label',\n"
                    "{indent}        'size': len(_last_vals),\n"
                    "{indent}        'type': 'categorical',\n"
                    "{indent}    }\n"
                )

                def _build_replacement(m):
                    indent = m.group(1)
                    block = _inject.replace("{indent}", indent)
                    return block + m.group(0)

                modified = re.sub(
                    r'^([ \t]*)json_SOS_final\s*=\s*\{',
                    _build_replacement,
                    modified,
                    count=1,
                    flags=re.MULTILINE,
                )

            # ----------------------------------------------------------------
            # Fix 2 – belt-and-suspenders: in data_GOIO() iterate only over
            # feature columns (exclude the last "label" entry from the loop).
            # This catches any edge case where Fix 1 didn't fire.
            # ----------------------------------------------------------------
            modified = re.sub(
                r'([ \t]*)for\s+i\s*,\s*ds\s+in\s+enumerate\(data_info\[.columns.\]\)\s*:',
                lambda m: m.group(1) + "for i, ds in enumerate(data_info['columns'][:-1]):  # exclude target col",
                modified,
                count=1,
            )

            if modified != content:
                with open(dp_path, 'w', encoding='utf-8') as f:
                    f.write(modified)
                print("  Patched data_description/data_GOIO in dataprocessing.py")
            else:
                print("  dataprocessing.py already patched (skipping)")
        except Exception as e:
            print(f"  Warning: could not patch dataprocessing.py: {e}")

    print("  Patching complete")
    return repo_dir


def prepare_goio_dataset(dataset_name, X_train, y_train, repo_dir):
    """Prepare dataset in GOIO's expected format with info.json.

    The saved CSV must contain exactly n_features numeric (float) feature
    columns followed by one integer target column – the target MUST NOT be
    appended to X.  info.json records n_features == X_train.shape[1] so
    that GOIO's data pipeline never mistakes the target for a feature.
    """

    data_dir = f"{repo_dir}/data/datasets/{dataset_name}"
    os.makedirs(data_dir, exist_ok=True)

    # n_features is the number of FEATURE columns – target is separate.
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # Build the DataFrame with explicit dtypes:
    #   - feature columns: float64  (no mixed/object dtype)
    #   - target column  : int64    (integer class labels)
    col_names = [f'num_{i}' for i in range(n_features)]
    X_float = np.asarray(X_train, dtype=np.float64)
    y_int = np.asarray(y_train, dtype=np.int64)

    df_features = pd.DataFrame(X_float, columns=col_names)
    df_target = pd.Series(y_int, name="target")

    # Concatenate: target is a SEPARATE series, NOT appended to X.
    df = pd.concat([df_features, df_target], axis=1)

    csv_path = f"{data_dir}/{dataset_name}.csv"
    df.to_csv(csv_path, index=False)

    # Assertions: CSV must have exactly n_features+1 columns with the
    # target as the very last column, and n_features must match X.shape[1].
    _saved = pd.read_csv(csv_path)
    assert _saved.shape[1] == n_features + 1, (
        f"CSV column count mismatch: got {_saved.shape[1]}, "
        f"expected {n_features + 1} (n_features={n_features})"
    )
    assert _saved.columns[-1] == "target", (
        f"Last CSV column must be 'target', got '{_saved.columns[-1]}'"
    )
    assert n_features == X_train.shape[1], (
        f"n_features ({n_features}) != X_train.shape[1] ({X_train.shape[1]})"
    )

    # Create info.json for GOIO's MLVAE/CLDM training scripts.
    # n_features == X.shape[1] (features only, NOT including target).
    info = {
        "name": dataset_name,
        "task_type": "binclass" if n_classes == 2 else "multiclass",
        "n_num_features": n_features,
        "n_features": n_features,
        "n_cat_features": 0,
        "train_size": len(X_train),
        "val_size": 0,
        "test_size": 0,
        "n_classes": n_classes if n_classes > 2 else 0,
    }

    with open(f"{data_dir}/info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"      ✓ Saved CSV: {csv_path}")
    print(f"      ✓ Saved info.json: {n_features} features, {n_classes} classes")
    return n_features


def run_goio_command(repo_dir, dataname, method, mode, exp=0):
    """Execute GOIO main.py command"""
    
    cmd = [
        sys.executable,
        "main.py",
        "--dataname", dataname,
        "--method", method,
        "--mode", mode,
        "--exp", str(exp)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=1800
        )
        
        if result.returncode != 0:
            if result.stderr:
                print(f"        stderr: {result.stderr[-300:]}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"        Timeout")
        return False
    except Exception as e:
        print(f"        Error: {e}")
        return False


def train_goio_pipeline(dataset_name, repo_dir):
    """Run full GOIO pipeline"""
    
    # Step 1: Data preprocessing
    print(f"      [1/4] Data split...")
    if not run_goio_command(repo_dir, dataset_name, "data", "split", 0):
        print(f"        ✗ Failed")
        return False
    print(f"        ✓ Done")
    
    # Step 2: Train MLVAE
    print(f"      [2/4] MLVAE train...")
    if not run_goio_command(repo_dir, dataset_name, "MLVAE", "train", 0):
        print(f"        ✗ Failed")
        return False
    print(f"        ✓ Done")
    
    # Step 3: Train CLDM
    print(f"      [3/4] CLDM train...")
    if not run_goio_command(repo_dir, dataset_name, "CLDM", "train", 0):
        print(f"        ✗ Failed")
        return False
    print(f"        ✓ Done")
    
    # Step 4: Sample from CLDM
    print(f"      [4/4] CLDM sample...")
    if not run_goio_command(repo_dir, dataset_name, "CLDM", "sample", 0):
        print(f"        ✗ Failed")
        return False
    print(f"        ✓ Done")
    
    return True


def collect_goio_samples(dataset_name, repo_dir, n_features):
    """Collect synthetic samples from GOIO output"""
    
    # GOIO saves to: synthetic/{dataname}/exp{fold}/CLDM/syn_{dataname}.csv
    syn_path = f"{repo_dir}/synthetic/{dataset_name}/exp0/CLDM/syn_{dataset_name}.csv"
    
    if os.path.exists(syn_path):
        try:
            df = pd.read_csv(syn_path)
            
            print(f"        CSV shape: {df.shape}, expected features: {n_features}")
            
            # Strategy: GOIO outputs either n_features or n_features+1 columns
            # If n_features+1, the last column is the target (to be excluded)
            
            num_cols = []
            
            # Pattern 1: Columns named 'num_0', 'num_1', etc.
            num_cols = [c for c in df.columns if c.startswith('num_')]
            if len(num_cols) > 0:
                num_cols = sorted(num_cols, key=lambda x: int(x.split('_')[1]))[:n_features]
            
            # Pattern 2: Numeric string columns '0', '1', '2', etc.
            if len(num_cols) == 0:
                all_numeric = [c for c in df.columns if str(c).isdigit()]
                if len(all_numeric) > 0:
                    all_numeric = sorted(all_numeric, key=int)
                    
                    # If we have exactly n_features, use all
                    # If we have n_features+1, exclude last (target column)
                    if len(all_numeric) == n_features:
                        num_cols = all_numeric
                    elif len(all_numeric) == n_features + 1:
                        num_cols = all_numeric[:n_features]
                        print(f"        Excluding target column '{all_numeric[-1]}'")
                    else:
                        # Take first n_features
                        num_cols = all_numeric[:n_features]
            
            # Pattern 3: Named columns excluding known target names
            if len(num_cols) == 0:
                target_names = ['target', 'label', 'class', 'y', 'Target', 'Label', 'Class', 'Y']
                feature_cols = [c for c in df.columns if c not in target_names]
                
                if len(feature_cols) == n_features:
                    num_cols = feature_cols
                elif len(feature_cols) == n_features + 1:
                    # Assume last column is target
                    num_cols = feature_cols[:n_features]
                else:
                    num_cols = feature_cols[:n_features]
            
            # Validation
            if len(num_cols) == 0:
                print(f"        ✗ Could not identify feature columns")
                return None
            
            if len(num_cols) != n_features:
                print(f"        ⚠ Using {len(num_cols)} columns instead of expected {n_features}")
            
            # Extract features
            X_syn = df[num_cols].values.astype(np.float32)
            
            # Handle dimension mismatch
            if X_syn.shape[1] < n_features:
                # Pad with zeros if we have fewer features
                padding = np.zeros((len(X_syn), n_features - X_syn.shape[1]), dtype=np.float32)
                X_syn = np.hstack([X_syn, padding])
                print(f"        ⚠ Padded from {len(num_cols)} to {n_features} features")
            elif X_syn.shape[1] > n_features:
                # Trim if we have more features
                X_syn = X_syn[:, :n_features]
                print(f"        ⚠ Trimmed from {len(num_cols)} to {n_features} features")
            
            print(f"        ✓ Loaded {len(X_syn)} samples × {X_syn.shape[1]} features")
            return X_syn
            
        except Exception as e:
            print(f"        ✗ Error reading CSV: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"        ✗ File not found: {syn_path}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--X_train', type=str, required=True)
    parser.add_argument('--y_train', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    print(f"\nGOIO: {args.dataset}")
    print(f"{'='*60}")
    
    # Setup repository
    repo_dir = setup_goio_repo()
    if repo_dir is None:
        sys.exit(1)
    
    # Load data
    X_train = np.load(args.X_train)
    y_train = np.load(args.y_train)
    n_features = X_train.shape[1]
    
    print(f"  Data: {X_train.shape}, Classes: {np.unique(y_train)}")
    
    # Prepare dataset
    print(f"\n  Preparing...")
    prepare_goio_dataset(args.dataset, X_train, y_train, repo_dir)
    
    # Train pipeline
    print(f"\n  Training...")
    success = train_goio_pipeline(args.dataset, repo_dir)
    
    if not success:
        print(f"\n  ✗ GOIO Training Failed")
        sys.exit(1)
    
    # Collect samples
    print(f"\n  Collecting...")
    synthetic = collect_goio_samples(args.dataset, repo_dir, n_features)
    
    if synthetic is None or len(synthetic) == 0:
        print(f"\n  ✗ No samples generated")
        sys.exit(1)
    
    # Save results
    output_dir = f"results_goio/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/synthetic_samples.npy"
    
    np.save(output_path, synthetic)
    print(f"\n  ✓ Saved {len(synthetic)} samples")
    print(f"  ✓ Location: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
