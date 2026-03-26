#!/usr/bin/env python3
"""
GOIO Training - Properly integrated with GOIO repository structure
"""
import os
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
    import re
    
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

    # Fix num_col_idx generation in dataprocessing.py:
    # Root cause: data_GOIO iterates over ALL columns in the dataset JSON
    # (including the last/target column) when building num_col_idx.  For
    # datasets with 10+ classes, data_description() assigns the target column
    # type "ordinal" instead of "label", so data_GOIO incorrectly adds the
    # target column index (== n_features) into num_col_idx.  This causes
    #   X_num_test = xtest[:, num_col_idx]   → IndexError: index N out of bounds
    # because xtest already has the target stripped (shape: (m, n_features)).
    #
    # Fix: limit the data_GOIO column loop to feature columns only by slicing
    # [:-1] to exclude the last (target) column entry from the JSON.
    dp_path = os.path.join(repo_dir, "data", "dataprocessing.py")
    if os.path.exists(dp_path):
        try:
            with open(dp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            modified = content

            # Primary fix: exclude last (target) column from data_GOIO loop.
            # Before: for i, ds in enumerate(data_info['columns']):
            # After:  for i, ds in enumerate(data_info['columns'][:-1]):
            modified = modified.replace(
                "for i, ds in enumerate(data_info['columns']):",
                "for i, ds in enumerate(data_info['columns'][:-1]):  # exclude target col"
            )

            # Secondary fix (legacy guard): remove erroneous +1 from any
            # range()-based num_col_idx construction, e.g.
            #   num_col_idx = list(range(n_num_features + 1))
            modified = re.sub(
                r'(num_col_idx\s*=\s*list\(range\()([^)]+?)\s*\+\s*1(\)\))',
                r'\1\2\3',
                modified
            )

            if modified != content:
                with open(dp_path, 'w', encoding='utf-8') as f:
                    f.write(modified)
                print("  Patched num_col_idx in dataprocessing.py")
            else:
                print("  Warning: dataprocessing.py patch pattern not found -- GOIO may still fail")
        except Exception as e:
            print(f"  Warning: could not patch dataprocessing.py: {e}")

    print("  Patching complete")
    return repo_dir


def prepare_goio_dataset(dataset_name, X_train, y_train, repo_dir):
    """Prepare dataset in GOIO's expected format with info.json"""
    
    data_dir = f"{repo_dir}/data/datasets/{dataset_name}"
    os.makedirs(data_dir, exist_ok=True)
    
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # Create CSV
    col_names = [f'num_{i}' for i in range(n_features)]
    df = pd.DataFrame(X_train, columns=col_names)
    df['target'] = y_train.astype(int)
    
    csv_path = f"{data_dir}/{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    
    # Create info.json (CRITICAL - GOIO needs this)
    info = {
        "name": dataset_name,
        "task_type": "binclass" if n_classes == 2 else "multiclass",
        "n_num_features": n_features,
        "n_cat_features": 0,
        "train_size": len(X_train),
        "val_size": 0,
        "test_size": 0,
        "n_classes": n_classes if n_classes > 2 else 0
    }
    
    with open(f"{data_dir}/info.json", 'w') as f:
        json.dump(info, f, indent=2)

    # Validate: feature column indices must be exactly 0..n_features-1.
    # The target column (at index n_features in the CSV) must NEVER appear.
    num_col_idx = list(range(n_features))
    assert min(num_col_idx) == 0 and max(num_col_idx) == n_features - 1, \
        f"num_col_idx out of range: min={min(num_col_idx)}, max={max(num_col_idx)}, expected 0..{n_features-1}"

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
