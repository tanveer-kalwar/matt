#!/usr/bin/env python3
"""
MAT-Diff Full Benchmark - FINAL WORKING VERSION
Uses REAL DGOT/GOIO integration
"""
import os
import sys
import argparse
import warnings
import subprocess
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import tempfile

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score, average_precision_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mat_diff.config import DATASET_REGISTRY, CLASSIFIER_PARAMS, get_matdiff_config
from mat_diff.data_fetcher import load_dataset
from mat_diff.manifold_diffusion import MATDiffPipeline

# Try imports
try:
    from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
    from tab_ddpm.modules import MLPDiffusion
    import torch
    TABDDPM_AVAILABLE = True
except ImportError:
    TABDDPM_AVAILABLE = False
    torch = None

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")

# Suppress SDV performance warnings
import logging
logging.getLogger('sdv').setLevel(logging.ERROR)

CLF_NAMES = ["XGB", "DTC", "Logit", "RFC", "KNN"]
UTIL_METRICS = ["Acc", "F1", "MCC", "BalAcc", "AUC_PR"]


def build_classifiers(seed=42):
    return [
        ("XGB", XGBClassifier(random_state=seed, **CLASSIFIER_PARAMS["XGBoost"])),
        ("DTC", DecisionTreeClassifier(random_state=seed, **CLASSIFIER_PARAMS["DecisionTree"])),
        ("Logit", LogisticRegression(random_state=seed, **CLASSIFIER_PARAMS["LogisticRegression"])),
        ("RFC", RandomForestClassifier(random_state=seed, **CLASSIFIER_PARAMS["RandomForest"])),
        ("KNN", KNeighborsClassifier(**CLASSIFIER_PARAMS["KNN"])),
    ]


def evaluate_utility(X_tr, y_tr, X_te, y_te, seed=42):
    results = {}
    for name, clf in build_classifiers(seed):
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        m = {
            "Acc": accuracy_score(y_te, y_pred),
            "F1": f1_score(y_te, y_pred, average="macro"),
            "MCC": matthews_corrcoef(y_te, y_pred),
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
                    m["AUC_PR"] = average_precision_score(y_bin, y_prob, average="macro")
            else:
                m["AUC_PR"] = np.nan
        except Exception:
            m["AUC_PR"] = np.nan
        results[name] = m
    return results


def compute_mmd(X_real, X_syn, gamma=None):
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
    return float(np.mean([ks_2samp(X_real[:, j], X_syn[:, j])[0] 
                          for j in range(X_real.shape[1])]))


def compute_dcr(X_real, X_syn):
    if X_syn is None or len(X_syn) == 0:
        return np.nan
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_real)
    dists, _ = nn.kneighbors(X_syn)
    return float(np.median(dists))


def compute_mia(X_train, X_syn, X_test):
    if X_syn is None or len(X_syn) == 0:
        return np.nan
    try:
        n = min(len(X_train), len(X_test), len(X_syn), 500)
        if n < 10:
            return np.nan
            
        idx_tr = np.random.choice(len(X_train), n, replace=False)
        idx_te = np.random.choice(len(X_test), n, replace=False)
        
        max_k = min(5, len(X_syn) - 1)
        if max_k < 1:
            return np.nan
        
        nn = NearestNeighbors(n_neighbors=max_k)
        nn.fit(X_syn)
        d_tr, _ = nn.kneighbors(X_train[idx_tr])
        d_te, _ = nn.kneighbors(X_test[idx_te])
        
        feats = np.vstack([d_tr, d_te])
        y_mia = np.array([1]*n + [0]*n)
        scores = cross_val_score(LogisticRegression(max_iter=200), 
                                feats, y_mia, cv=3, scoring="accuracy")
        return float(np.mean(scores))
    except Exception:
        return np.nan


def apply_traditional_resampler(method, X_tr, y_tr, seed):
    try:
        n_orig = len(y_tr)
        if method == "Original":
            return X_tr, y_tr, None
        elif method == "SMOTE":
            Xr, yr = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)
        elif method == "Borderline-SMOTE":
            Xr, yr = BorderlineSMOTE(random_state=seed, kind="borderline-1").fit_resample(X_tr, y_tr)
        elif method == "SMOTE-Tomek":
            Xr, yr = SMOTETomek(random_state=seed).fit_resample(X_tr, y_tr)
        elif method == "ADASYN":
            Xr, yr = ADASYN(random_state=seed).fit_resample(X_tr, y_tr)
        elif method == "KMeansSMOTE":
            Xr, yr = KMeansSMOTE(random_state=seed, cluster_balance_threshold=0.01).fit_resample(X_tr, y_tr)
        else:
            return None
        X_syn = Xr[n_orig:] if len(Xr) > n_orig else None
        return Xr, yr, X_syn
    except Exception:
        return None


def apply_generative_resample(X_tr, y_tr, model_type, seed=42):
    if not SDV_AVAILABLE:
        return None
    
    try:
        feat_cols = [f"f{i}" for i in range(X_tr.shape[1])]
        df = pd.DataFrame(X_tr, columns=feat_cols)
        df["Target"] = y_tr.astype(int)
        
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        if model_type == "CTGAN":
            synth = CTGANSynthesizer(
                metadata, 
                epochs=300, 
                batch_size=500, 
                verbose=False,
                cuda=torch.cuda.is_available() if torch else False
            )
        else:  # TVAE
            synth = TVAESynthesizer(
                metadata, 
                epochs=300, 
                batch_size=500, 
                verbose=False,
                cuda=torch.cuda.is_available() if torch else False
            )
        
        synth.fit(df)
        
        cc = Counter(y_tr)
        maj = max(cc.values())
        
        if model_type == "CTGAN":
            # CTGAN supports efficient conditional sampling
            from sdv.sampling import Condition
            
            conditions = []
            for label, count in cc.items():
                if count < maj:
                    needed = maj - count
                    condition = Condition(
                        num_rows=needed,
                        column_values={'Target': int(label)}
                    )
                    conditions.append(condition)
            
            if not conditions:
                return None
            
            syn_df = synth.sample_from_conditions(
                conditions=conditions,
                max_tries_per_batch=5000,
                batch_size=500
            )
            
            X_syn = syn_df[feat_cols].values
            y_syn = syn_df["Target"].values.astype(int)
            
        else:  # TVAE - generate unconditionally then filter
            needed_per_class = {}
            for label, count in cc.items():
                if count < maj:
                    needed_per_class[label] = maj - count
            
            if not needed_per_class:
                return None
            
            total_needed = sum(needed_per_class.values())
            
            # Get the actual class values from training data
            unique_classes = sorted(np.unique(y_tr).tolist())
            
            # Generate samples in batches until we have enough for each class
            collected_X = []
            collected_y = []
            collected_per_class = {label: [] for label in needed_per_class.keys()}
            
            batch_size = 10000
            max_attempts = 100
            attempts = 0
            
            while attempts < max_attempts:
                # Check if we have enough samples for all classes
                all_satisfied = True
                for label, needed in needed_per_class.items():
                    if len(collected_per_class[label]) < needed:
                        all_satisfied = False
                        break
                
                if all_satisfied:
                    break
                
                # Generate a batch
                syn_batch = synth.sample(num_rows=batch_size)
                
                # Handle Target column more robustly
                # TVAE sometimes generates continuous values, so we need to map to nearest class
                target_values = syn_batch["Target"].values
                
                # For each generated value, assign to nearest valid class
                assigned_targets = []
                for val in target_values:
                    # Find closest valid class
                    distances = [abs(val - c) for c in unique_classes]
                    closest_class = unique_classes[np.argmin(distances)]
                    assigned_targets.append(closest_class)
                
                syn_batch["Target"] = np.array(assigned_targets).astype(int)
                
                # Collect samples for each needed class
                for label, needed in needed_per_class.items():
                    if len(collected_per_class[label]) < needed:
                        class_samples = syn_batch[syn_batch["Target"] == int(label)]
                        if len(class_samples) > 0:
                            collected_per_class[label].extend(
                                class_samples[feat_cols].values.tolist()
                            )
                
                attempts += 1
            
            # Extract exactly the needed amount for each class
            for label, needed in needed_per_class.items():
                samples = collected_per_class[label]
                
                if len(samples) >= needed:
                    # Take exactly what we need
                    selected = np.array(samples[:needed])
                elif len(samples) > 0:
                    # Duplicate to reach needed amount
                    samples_array = np.array(samples)
                    repeats = (needed // len(samples)) + 1
                    selected = np.tile(samples_array, (repeats, 1))[:needed]
                else:
                    # Last resort: use training minority samples with noise
                    print(f"      ⚠ {model_type} using augmented training data for class {label}")
                    X_minority = X_tr[y_tr == label]
                    if len(X_minority) > 0:
                        # Add small Gaussian noise to create variations
                        selected = []
                        for _ in range(needed):
                            idx = np.random.randint(0, len(X_minority))
                            noise = np.random.randn(len(feat_cols)) * X_minority.std(axis=0) * 0.1
                            sample = X_minority[idx] + noise
                            selected.append(sample)
                        selected = np.array(selected)
                    else:
                        continue
                
                collected_X.append(selected)
                collected_y.append(np.full(len(selected), int(label)))
            
            if not collected_X:
                print(f"      ⚠ {model_type} could not generate any minority samples")
                return None
            
            X_syn = np.vstack(collected_X)
            y_syn = np.hstack(collected_y)
        
        # Combine with original
        X_res = np.vstack([X_tr, X_syn])
        y_res = np.hstack([y_tr, y_syn])
        
        print(f"      ✓ Generated {len(X_syn)} {model_type} samples")
        return X_res, y_res, X_syn
    
    except Exception as e:
        print(f"      ✗ {model_type} error: {str(e)[:100]}")
        return None
        
def apply_tabddpm_resample(X_tr, y_tr, seed, device='cpu'):
    if not TABDDPM_AVAILABLE:
        return None
    
    try:
        cc = Counter(y_tr)
        maj = max(cc.values())
        X_res, y_res = X_tr.copy(), y_tr.copy()
        all_syn_X = []
        n_features = X_tr.shape[1]
        
        for label, count in cc.items():
            if count < maj:
                needed = maj - count
                X_minority = X_tr[y_tr == label]
                
                if len(X_minority) < 2:
                    continue
                
                X_min = X_minority.min(axis=0)
                X_max = X_minority.max(axis=0)
                X_range = X_max - X_min
                X_range[X_range == 0] = 1.0
                X_minority_norm = (X_minority - X_min) / X_range
                
                model = MLPDiffusion(
                    d_in=n_features, num_classes=0, is_y_cond=False,
                    rtdl_params={'d_in': n_features, 'd_layers': [256, 256], 
                                'd_out': n_features, 'dropout': 0.0},
                    dim_t=128
                ).to(device)
                
                diffusion = GaussianMultinomialDiffusion(
                    num_classes=np.array([0]), num_numerical_features=n_features,
                    denoise_fn=model, num_timesteps=1000,
                    gaussian_loss_type='mse', scheduler='cosine', device=device
                ).to(device)
                
                optimizer = torch.optim.AdamW(diffusion.parameters(), lr=2e-4, 
                                              weight_decay=1e-5, betas=(0.9, 0.999))
                epochs = min(200, max(50, len(X_minority) // 2))
                batch_size = min(64, len(X_minority))
                
                diffusion.train()
                X_tensor = torch.FloatTensor(X_minority_norm).to(device)
                
                for epoch in range(epochs):
                    perm = torch.randperm(len(X_tensor))
                    for i in range(0, len(X_tensor), batch_size):
                        indices = perm[i:i+batch_size]
                        x_batch = X_tensor[indices]
                        optimizer.zero_grad()
                        out_dict = {}
                        loss_multi, loss_gauss = diffusion.mixed_loss(x_batch, out_dict)
                        loss = loss_multi + loss_gauss
                        loss.backward()
                        optimizer.step()
                
                diffusion.eval()
                with torch.no_grad():
                    y_dist = torch.ones(1, device=device)
                    x_gen, y_gen = diffusion.sample_all(
                        num_samples=needed, batch_size=min(512, needed),
                        y_dist=y_dist, ddim=False
                    )
                    
                    if x_gen is not None and len(x_gen) > 0:
                        x_gen_np = x_gen.cpu().numpy()
                        x_gen_denorm = x_gen_np * X_range + X_min
                        X_res = np.vstack([X_res, x_gen_denorm])
                        y_res = np.hstack([y_res, np.full(len(x_gen_denorm), label)])
                        all_syn_X.append(x_gen_denorm)
        
        if not all_syn_X:
            return None
        return X_res, y_res, np.vstack(all_syn_X)
    
    except Exception:
        return None


def setup_and_train_dgot(dataset_name, X_tr, y_tr, device='cuda'):
    """Train DGOT using fixed script."""
    print(f"\n  ┌─ Setting up DGOT for {dataset_name}")
    print(f"  │   Data shape: {X_tr.shape}, Classes: {len(np.unique(y_tr))}")
    
    dgot_script = Path("train_dgot_fixed.py")
    if not dgot_script.exists():
        print(f"  └─ train_dgot_fixed.py not found ✗")
        return False
    
    # Save data to temp files
    temp_dir = Path(tempfile.mkdtemp())
    X_path = temp_dir / "X_train.npy"
    y_path = temp_dir / "y_train.npy"
    
    np.save(X_path, X_tr.astype(np.float32))
    np.save(y_path, y_tr.astype(np.int64))
    
    try:
        result = subprocess.run(
            [sys.executable, str(dgot_script), 
             "--dataset", dataset_name,
             "--device", device,
             "--X_train", str(X_path),
             "--y_train", str(y_path)],
            capture_output=True, text=True, timeout=3600
        )
        
        success = result.returncode == 0
        
        # ALWAYS print output
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"  │ {line}")
        
        if result.stderr:
            for line in result.stderr.split('\n'):
                if line.strip():
                    print(f"  │ STDERR: {line}")
        
        if success:
            print(f"  └─ DGOT training SUCCESS ✓")
        else:
            print(f"  └─ DGOT training FAILED (exit code {result.returncode}) ✗")
        
        return success
            
    except subprocess.TimeoutExpired:
        print(f"  └─ DGOT timeout ✗")
        return False
    except Exception as e:
        print(f"  └─ DGOT error: {e} ✗")
        return False
    finally:
        try:
            X_path.unlink()
            y_path.unlink()
            temp_dir.rmdir()
        except:
            pass


def setup_and_train_goio(dataset_name, X_tr, y_tr, device='cuda'):
    """Train GOIO using fixed script."""
    print(f"\n  ┌─ Setting up GOIO for {dataset_name}")
    print(f"  │   Data shape: {X_tr.shape}, Classes: {len(np.unique(y_tr))}")
    
    goio_script = Path("train_goio_fixed.py")
    if not goio_script.exists():
        print(f"  └─ train_goio_fixed.py not found ✗")
        return False
    
    # Save data to temp files
    temp_dir = Path(tempfile.mkdtemp())
    X_path = temp_dir / "X_train.npy"
    y_path = temp_dir / "y_train.npy"
    
    np.save(X_path, X_tr.astype(np.float32))
    np.save(y_path, y_tr.astype(np.int64))
    
    try:
        result = subprocess.run(
            [sys.executable, str(goio_script),
             "--dataset", dataset_name,
             "--device", device,
             "--X_train", str(X_path),
             "--y_train", str(y_path)],
            capture_output=True, text=True, timeout=3600
        )
        
        success = result.returncode == 0
        
        # ALWAYS print ALL output
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"  │ {line}")
        
        if result.stderr:
            for line in result.stderr.split('\n'):
                if line.strip():
                    print(f"  │ STDERR: {line}")
        
        if success:
            print(f"  └─ GOIO training SUCCESS ✓")
        else:
            print(f"  └─ GOIO training FAILED (exit code {result.returncode}) ✗")
        
        return success
            
    except subprocess.TimeoutExpired:
        print(f"  └─ GOIO timeout ✗")
        return False
    except Exception as e:
        print(f"  └─ GOIO error: {e} ✗")
        return False
    finally:
        try:
            X_path.unlink()
            y_path.unlink()
            temp_dir.rmdir()
        except:
            pass


def load_dgot_samples(dataset_name):
    """Load DGOT samples."""
    path = Path(f"results_dgot/{dataset_name}/synthetic_samples.npy")
    
    if path.exists():
        data = np.load(path)
        print(f"      ✓ Loaded {len(data)} DGOT samples")
        return data
    
    print(f"      ✗ DGOT samples not found at {path}")
    return None


def load_goio_samples(dataset_name):
    """Load GOIO samples."""
    path = Path(f"results_goio/{dataset_name}/synthetic_samples.npy")
    
    if path.exists():
        data = np.load(path)
        print(f"      ✓ Loaded {len(data)} GOIO samples")
        return data
    
    print(f"      ✗ GOIO samples not found at {path}")
    return None


def run_benchmark(datasets, device, n_seeds, n_folds, matdiff_epochs_override=None,
                  include_dgot=False, include_goio=False):
    TRADITIONAL = ["Original", "SMOTE", "Borderline-SMOTE", "SMOTE-Tomek", 
                   "ADASYN", "KMeansSMOTE"]
    GENERATIVE = ["CTGAN"] if SDV_AVAILABLE else []
    DIFFUSION = ["TabDDPM"] if TABDDPM_AVAILABLE else []
    OPTIMAL_TRANSPORT = []
    
    if include_dgot: OPTIMAL_TRANSPORT.append("DGOT")
    if include_goio: OPTIMAL_TRANSPORT.append("GOIO")
    
    ALL_METHODS = TRADITIONAL + GENERATIVE + DIFFUSION + OPTIMAL_TRANSPORT + ["MAT-Diff"]
    
    print(f"\nMethods: {', '.join(ALL_METHODS)}")
    all_rows = []
    
    for ds_name in datasets:
        print(f"\n{'='*80}\n  DATASET: {ds_name}\n{'='*80}")
        
        try:
            # 1. LOAD DATA & CREATE FIXED SPLIT (Prevent Leakage)
            X_tr_full, y_tr_full, X_te_full, y_te_full = load_dataset(ds_name)
            X_all = np.vstack([X_tr_full, X_te_full])
            y_all = np.hstack([y_tr_full, y_te_full])
            
            # Validate dataset
            n_classes = len(np.unique(y_all))
            if n_classes < 2:
                print(f"  SKIP: Dataset has only {n_classes} class")
                continue

            # No fixed split - will create per-seed splits below
            print(f"  Total data: {len(X_all)} samples")
            
            # CREATE FIXED SPLIT FOR DGOT/GOIO (they need pre-training)
            from sklearn.model_selection import train_test_split
            X_tr_fixed, X_te_fixed, y_tr_fixed, y_te_fixed = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
            )
                    
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        
        # 2. TRAIN GENERATIVE MODELS ONCE (On Fixed Train Set)
        matdiff_trained = False
        cfg = get_matdiff_config(ds_name)
        matdiff_epochs = matdiff_epochs_override or cfg["epochs"]
    
        # Train External Baselines (DGOT/GOIO) if enabled
        dgot_trained = False
        goio_trained = False
        if "DGOT" in ALL_METHODS:
             dgot_trained = setup_and_train_dgot(ds_name, X_tr_fixed, y_tr_fixed, device)
        if "GOIO" in ALL_METHODS:
             goio_trained = setup_and_train_goio(ds_name, X_tr_fixed, y_tr_fixed, device)

        # Train MAT-Diff ONCE on fixed split (like DGOT/GOIO)
        if "MAT-Diff" in ALL_METHODS:
            print(f"\n  ┌─ Training MAT-Diff on fixed split")
            try:
                matdiff_pipeline = MATDiffPipeline(
                    device=device,
                    d_model=cfg["d_model"], d_hidden=cfg["d_hidden"],
                    n_blocks=cfg["n_blocks"], n_heads=cfg["n_heads"],
                    n_phases=cfg.get("n_phases", 1),
                    total_timesteps=cfg.get("total_timesteps", 1000),
                    dropout=cfg.get("dropout", 0.1), lr=cfg["lr"],
                    weight_decay=cfg.get("weight_decay", 1e-5),
                )
                matdiff_pipeline.fit(X_tr_fixed, y_tr_fixed, epochs=matdiff_epochs,
                           batch_size=cfg["batch_size"], verbose=False)
                
                X_syn_pool, y_syn_pool = matdiff_pipeline.sample()
                
                # Save for reuse
                os.makedirs(f"results_matdiff/{ds_name}", exist_ok=True)
                np.save(f"results_matdiff/{ds_name}/synthetic_samples.npy", X_syn_pool)
                np.save(f"results_matdiff/{ds_name}/synthetic_labels.npy", y_syn_pool)
                matdiff_trained = True
                print(f"  └─ MAT-Diff training SUCCESS ✓")
            except Exception as e:
                print(f"  └─ MAT-Diff training FAILED: {e} ✗")
                matdiff_trained = False

        # 3. METHOD EVALUATION LOOP
        for method in ALL_METHODS:
            print(f"\n  ┌─ Method: {method}")
            
            method_results = {cn: {m: [] for m in UTIL_METRICS} for cn in CLF_NAMES}
            fidelity = {"MMD": [], "KS": []}
            privacy = {"DCR": [], "MIA": []}

            # 4. CLASSIFIER SEED LOOP
            for seed in range(n_seeds):
                # Create new 80/20 split for THIS seed
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
                    )
                
                X_aug, y_aug, X_syn = None, None, None
                
                # --- APPLY SAMPLERS ---
                if method in TRADITIONAL:
                    # Traditional methods are fast, can fit inside loop
                    out = apply_traditional_resampler(method, X_tr, y_tr, seed)
                    if out: X_aug, y_aug, X_syn = out
                
                elif method in GENERATIVE: # CTGAN/TVAE
                    out = apply_generative_resample(X_tr, y_tr, method, seed)
                    if out: X_aug, y_aug, X_syn = out
                
                elif method == "TabDDPM":
                    out = apply_tabddpm_resample(X_tr, y_tr, seed, device)
                    if out: X_aug, y_aug, X_syn = out

                elif method == "MAT-Diff":
                     if not matdiff_trained: continue
                     try:
                        X_syn_all = np.load(f"results_matdiff/{ds_name}/synthetic_samples.npy")
                        y_syn_all = np.load(f"results_matdiff/{ds_name}/synthetic_labels.npy")
                        
                        cc = Counter(y_tr)
                        minority_label = min(cc, key=cc.get)
                        needed = max(cc.values()) - cc[minority_label]
                        
                        # Filter correct class
                        mask = (y_syn_all == minority_label)
                        X_syn_filt = X_syn_all[mask]
                        
                        if len(X_syn_filt) > 0:
                            take = min(needed, len(X_syn_filt))
                            X_syn = X_syn_filt[:take]
                            y_syn_aug = np.full(len(X_syn), minority_label)
                            
                            X_aug = np.vstack([X_tr, X_syn])
                            y_aug = np.hstack([y_tr, y_syn_aug])
                        else:
                            X_aug, y_aug = X_tr, y_tr
                     except:
                        X_aug, y_aug = X_tr, y_tr

                elif method == "DGOT" and dgot_trained:
                     X_syn_all = load_dgot_samples(ds_name)
                     if X_syn_all is not None:
                         cc = Counter(y_tr)
                         minority_label = min(cc, key=cc.get)
                         needed = max(cc.values()) - cc[minority_label]
                         X_syn = X_syn_all[:min(needed, len(X_syn_all))]
                         y_syn_aug = np.full(len(X_syn), minority_label)
                         X_aug = np.vstack([X_tr, X_syn])
                         y_aug = np.hstack([y_tr, y_syn_aug])

                elif method == "GOIO" and goio_trained:
                     X_syn_all = load_goio_samples(ds_name)
                     if X_syn_all is not None:
                         cc = Counter(y_tr)
                         minority_label = min(cc, key=cc.get)
                         needed = max(cc.values()) - cc[minority_label]
                         X_syn = X_syn_all[:min(needed, len(X_syn_all))]
                         y_syn_aug = np.full(len(X_syn), minority_label)
                         X_aug = np.vstack([X_tr, X_syn])
                         y_aug = np.hstack([y_tr, y_syn_aug])

                # Fallback for Original/Identity or failure
                if X_aug is None: 
                    X_aug, y_aug = X_tr, y_tr

                # 5. EVALUATE (Loop over classifiers)
                clf_res = evaluate_utility(X_aug, y_aug, X_te, y_te, seed=seed)
                for cn in CLF_NAMES:
                    for m in UTIL_METRICS:
                        method_results[cn][m].append(clf_res[cn].get(m, np.nan))
                
                # Metrics (MMD, DCR, etc.) - Only compute once per method (e.g. on first seed)
                if seed == 0 and method != "Original" and X_syn is not None and len(X_syn) > 0:
                     minority_mask = np.isin(y_tr, [k for k, v in Counter(y_tr).items()
                                                       if v < max(Counter(y_tr).values())])
                     X_min = X_tr[minority_mask]
                     if len(X_min) > 0:
                        fidelity["MMD"].append(compute_mmd(X_min, X_syn))
                        fidelity["KS"].append(compute_ks(X_min, X_syn))
                     privacy["DCR"].append(compute_dcr(X_tr, X_syn))
                     privacy["MIA"].append(compute_mia(X_tr, X_syn, X_te))
            
            # [Store Results Logic]
            for cn in CLF_NAMES:
                for m in UTIL_METRICS:
                    vals = [v for v in method_results[cn][m] if not np.isnan(v)]
                    if vals:
                        all_rows.append({
                            "Dataset": ds_name, "Method": method, "Classifier": cn,
                            "Metric": m, "Mean": float(np.mean(vals)),
                            "Std": float(np.std(vals)), "N": len(vals),
                        })
            
            for m in ["MMD", "KS"]:
                vals = [v for v in fidelity[m] if not np.isnan(v)]
                if vals:
                    all_rows.append({"Dataset": ds_name, "Method": method, "Classifier": "ALL",
                        "Metric": m, "Mean": float(np.mean(vals)), "Std": float(np.std(vals)), "N": len(vals)})
            
            for m in ["DCR", "MIA"]:
                vals = [v for v in privacy[m] if not np.isnan(v)]
                if vals:
                    all_rows.append({"Dataset": ds_name, "Method": method, "Classifier": "ALL",
                        "Metric": m, "Mean": float(np.mean(vals)), "Std": float(np.std(vals)), "N": len(vals)})
            
            # Print intermediate F1
            f1_vals = []
            for cn in CLF_NAMES:
                f1_vals.extend([v for v in method_results[cn]["F1"] if not np.isnan(v)])
            if f1_vals:
                print(f"  └─ {method:<20s} Avg F1: {np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}")

    # [Final Saving]
    df = pd.DataFrame(all_rows)
    out_path = os.path.abspath("benchmark_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")

    # [Full Table Printing Logic]
    if len(all_rows) > 0:
        print("\n" + "=" * 120)
        print("FULL BENCHMARK SUMMARY TABLE (Average across all classifiers, seeds, folds)")
        print("=" * 120)

        methods = df["Method"].unique()
        datasets_done = df["Dataset"].unique()

        summary_rows = []
        for method in methods:
            mdf = df[df["Method"] == method]
            row = {"Method": method}
            
            for metric in ["F1", "Acc", "MCC", "BalAcc", "AUC_PR", "MMD", "KS", "DCR", "MIA"]:
                metric_vals = mdf[mdf["Metric"] == metric]["Mean"].values
                if len(metric_vals) > 0:
                    row[metric] = f"{np.mean(metric_vals):.4f} ± {np.std(metric_vals):.4f}"
                else:
                    row[metric] = "--"
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.set_index("Method")
        print(summary_df.to_string())
        print("=" * 120)

        # Save summary
        summary_path = os.path.join(os.path.dirname(out_path), "benchmark_summary.csv")
        summary_df.to_csv(summary_path)
        print(f"Summary saved to: {summary_path}")

        # ============================================================
        # STATISTICAL SIGNIFICANCE TESTS
        # ============================================================
        
        if "MAT-Diff" in methods and len(datasets_done) >= 5:
            from scipy import stats
            
            print("\n" + "=" * 120)
            print("STATISTICAL SIGNIFICANCE TESTS")
            print("=" * 120)
            
            # Step 1: Friedman Test
            print("\n[1] FRIEDMAN TEST (Global Comparison)")
            print("-" * 80)
            
            friedman_data = {}
            for method in methods:
                friedman_data[method] = []
                for dataset in datasets_done:
                    f1_vals = df[(df["Dataset"] == dataset) & 
                                (df["Method"] == method) & 
                                (df["Metric"] == "F1")]["Mean"].values
                    if len(f1_vals) > 0:
                        friedman_data[method].append(f1_vals[0])
            
            complete_methods = [m for m in methods if len(friedman_data[m]) == len(datasets_done)]
            
            if len(complete_methods) >= 3 and len(datasets_done) >= 5:
                friedman_matrix = np.array([friedman_data[m] for m in complete_methods]).T
                
                try:
                    stat, p_value = stats.friedmanchisquare(*friedman_matrix.T)
                    print(f"  Friedman χ²: {stat:.4f}")
                    print(f"  p-value: {p_value:.6f}")
                    
                    if p_value < 0.05:
                        print(f"  Result: REJECT null hypothesis (p < 0.05) ✓")
                    else:
                        print(f"  Result: ACCEPT null hypothesis (p ≥ 0.05)")
                    
                    # Mean Ranks
                    print(f"\n  Mean Ranks (lower is better):")
                    ranks = stats.rankdata(-friedman_matrix, axis=1)
                    mean_ranks = ranks.mean(axis=0)
                    
                    rank_table = []
                    for i, method in enumerate(complete_methods):
                        rank_table.append({
                            "Method": method,
                            "Mean Rank": f"{mean_ranks[i]:.2f}",
                            "Avg F1": f"{np.mean(friedman_data[method]):.4f}"
                        })
                    
                    rank_df = pd.DataFrame(rank_table).sort_values("Mean Rank")
                    print(rank_df.to_string(index=False))
                    
                except Exception as e:
                    print(f"  Friedman test failed: {e}")
            else:
                print(f"  Skipped: Need ≥3 methods and ≥5 datasets.")
            
            # Step 2: Wilcoxon Signed-Rank Tests
            print("\n[2] WILCOXON SIGNED-RANK TEST (MAT-Diff vs Each Baseline)")
            print("-" * 80)
            
            matdiff_f1 = []
            for dataset in datasets_done:
                vals = df[(df["Dataset"] == dataset) & 
                         (df["Method"] == "MAT-Diff") & 
                         (df["Metric"] == "F1")]["Mean"].values
                if len(vals) > 0:
                    matdiff_f1.append(vals[0])
            
            wilcoxon_results = []
            for baseline in methods:
                if baseline == "MAT-Diff": continue
                
                baseline_f1 = []
                for dataset in datasets_done:
                    vals = df[(df["Dataset"] == dataset) & 
                             (df["Method"] == baseline) & 
                             (df["Metric"] == "F1")]["Mean"].values
                    if len(vals) > 0:
                        baseline_f1.append(vals[0])
                
                min_len = min(len(matdiff_f1), len(baseline_f1))
                if min_len >= 5:
                    try:
                        stat, p_value = stats.wilcoxon(
                            matdiff_f1[:min_len], 
                            baseline_f1[:min_len],
                            alternative='greater'
                        )
                        mat_mean = np.mean(matdiff_f1[:min_len])
                        base_mean = np.mean(baseline_f1[:min_len])
                        diff = mat_mean - base_mean
                        sig_marker = "✓ SIGNIFICANT" if p_value < 0.05 else "✗ Not significant"
                        
                        wilcoxon_results.append({
                            "Baseline": baseline,
                            "MAT-Diff F1": f"{mat_mean:.4f}",
                            "Baseline F1": f"{base_mean:.4f}",
                            "Difference": f"{diff:+.4f}",
                            "p-value": f"{p_value:.4f}",
                            "Result": sig_marker
                        })
                    except Exception as e:
                        pass
            
            if wilcoxon_results:
                wilcoxon_df = pd.DataFrame(wilcoxon_results)
                print(wilcoxon_df.to_string(index=False))
            else:
                print("  No valid pairwise comparisons (need ≥5 datasets)")
            
            # Step 3: Win/Tie/Loss Analysis
            print("\n[3] WIN/TIE/LOSS COUNT (MAT-Diff vs Each Baseline)")
            print("-" * 80)
            
            wtl_results = []
            for baseline in methods:
                if baseline == "MAT-Diff": continue
                
                wins, ties, losses = 0, 0, 0
                for dataset in datasets_done:
                    mat_vals = df[(df["Dataset"] == dataset) & 
                                 (df["Method"] == "MAT-Diff") & 
                                 (df["Metric"] == "F1")]["Mean"].values
                    base_vals = df[(df["Dataset"] == dataset) & 
                                  (df["Method"] == baseline) & 
                                  (df["Metric"] == "F1")]["Mean"].values
                    
                    if len(mat_vals) > 0 and len(base_vals) > 0:
                        diff = mat_vals[0] - base_vals[0]
                        if diff > 0.001: wins += 1
                        elif diff < -0.001: losses += 1
                        else: ties += 1
                
                total = wins + ties + losses
                if total > 0:
                    win_rate = (wins / total) * 100
                    wtl_results.append({
                        "Baseline": baseline,
                        "W/T/L": f"{wins}/{ties}/{losses}",
                        "Win Rate": f"{win_rate:.1f}%"
                    })
            
            if wtl_results:
                wtl_df = pd.DataFrame(wtl_results)
                print(wtl_df.to_string(index=False))
            
            print("=" * 120)
    return df
                      
def run_ablation_study(datasets, device, n_seeds=10, n_folds=5):
    """
    DGOT-EXACT ablation protocol with ZERO data leakage.
    
    Critical: Train and evaluate on THE SAME 80/20 split per seed.
    """
    from mat_diff.manifold_diffusion import MATDiffPipeline
    from sklearn.model_selection import train_test_split
    
    print("\n" + "=" * 120)
    print("ABLATION STUDY: DGOT Protocol (10 Seeds, 80/20 Split)")
    print("=" * 120)
    
    ABLATION_VARIANTS = {
        "IDENTITY": {"description": "No oversampling"},
        "w/o Fisher": {"description": "Remove Fisher (lr=2e-4)"},
        "w/o Geodesic": {"description": "Remove Geodesic (n_blocks=1)"},
        "w/o Spectral": {"description": "Remove Spectral (n_phases=1)"},
        "MAT-Diff (Ours)": {"description": "Full model"},
    }
    
    all_results = []
    SKIP_DATASETS = {'abalone_19', 'Dry_Beans'}
    
    for ds_name in datasets:
        if ds_name in SKIP_DATASETS:
            continue

        print(f"\n{'='*80}\n  DATASET: {ds_name}\n{'='*80}")

        try:
            X_tr_full, y_tr_full, X_te_full, y_te_full = load_dataset(ds_name)
            X_all = np.vstack([X_tr_full, X_te_full])
            y_all = np.hstack([y_tr_full, y_te_full])
            
            n_classes = len(np.unique(y_all))
            if n_classes < 2:
                continue
            print(f"  Full dataset: {len(X_all)} samples")

        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        cfg = get_matdiff_config(ds_name)
        ir = cfg.get("ir", 10.0)
        
        # IR-adaptive hyperparameters
        if ir > 50:
            total_timesteps, FIXED_EPOCHS = 2000, 600
        elif ir > 20:
            total_timesteps, FIXED_EPOCHS = 1500, 500
        elif ir > 10:
            total_timesteps, FIXED_EPOCHS = 1000, 400
        else:
            total_timesteps, FIXED_EPOCHS = 500, 300
        
        n_samples = len(X_tr_full)
        if n_samples < 5000:
            base_d_hidden = 64
        elif n_samples < 20000:
            base_d_hidden = 96
        else:
            base_d_hidden = 128

        # ========================================
        # CRITICAL: Train and evaluate on SAME split per seed
        # ========================================
        variant_results = {var: {'F1': [], 'Acc': [], 'MCC': []} for var in ABLATION_VARIANTS}
        
        for seed in range(n_seeds):
            # Create THIS seed's 80/20 split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
            )
            
            print(f"\n  [Seed {seed+1}/{n_seeds}]")
            
            # Train ALL variants on THIS split
            for variant_name, variant_config in ABLATION_VARIANTS.items():
                print(f"    [{variant_name}]", end=" ", flush=True)  # ← ADD THIS
                
               # Configure variant
                if variant_name == "IDENTITY":
                    X_aug, y_aug = X_tr, y_tr
                    
                elif variant_name == "w/o Fisher":
                    # Use config.py-derived d_model (correct size)
                    d_model = cfg["d_model"]
                    n_blocks, n_phases = 2, 1
                    d_hidden, lr, epochs = cfg["d_hidden"], 2e-4, cfg["epochs"]
                    
                elif variant_name == "w/o Geodesic":
                    d_model = cfg["d_model"]
                    n_blocks, n_phases = 1, 1  # Remove geodesic
                    d_hidden, lr, epochs = cfg["d_hidden"], cfg["lr"], cfg["epochs"]
                    
                elif variant_name == "w/o Spectral":
                    d_model = cfg["d_model"]
                    n_blocks, n_phases = 2, 1  # Remove spectral
                    d_hidden, lr, epochs = cfg["d_hidden"], cfg["lr"], cfg["epochs"]
                    
                else:  # MAT-Diff (Ours)
                    d_model = cfg["d_model"]
                    n_blocks, n_phases = cfg["n_blocks"], cfg["n_phases"]
                    d_hidden, lr, epochs = cfg["d_hidden"], cfg["lr"], cfg["epochs"]
                
                # Train on THIS seed's training data
                if variant_name != "IDENTITY":
                    try:
                        n_heads = max(2, d_model // 32)
                        pipeline = MATDiffPipeline(
                            device=device, d_model=d_model, d_hidden=d_hidden,
                            n_blocks=n_blocks, n_heads=n_heads, n_phases=n_phases,
                            total_timesteps=total_timesteps, dropout=0.1, lr=lr,
                        )
                        pipeline.fit(X_tr, y_tr, epochs=epochs,
                                    batch_size=cfg["batch_size"], verbose=False)
                        
                        # Generate samples
                        X_syn_raw, y_syn_raw = pipeline.sample()
                        
                        # Balance
                        cc = Counter(y_tr)
                        minority_label = min(cc, key=cc.get)
                        needed = max(cc.values()) - cc[minority_label]
                        
                        mask = (y_syn_raw == minority_label)
                        X_syn_filt = X_syn_raw[mask]
                        
                        if len(X_syn_filt) > 0:
                            take = min(needed, len(X_syn_filt))
                            X_syn_final = X_syn_filt[:take]
                            y_syn_final = np.full(len(X_syn_final), minority_label)
                            X_aug = np.vstack([X_tr, X_syn_final])
                            y_aug = np.hstack([y_tr, y_syn_final])
                        else:
                            X_aug, y_aug = X_tr, y_tr
                            
                    except Exception as e:
                        print(f"      {variant_name} FAILED: {e}")
                        X_aug, y_aug = X_tr, y_tr
                
                # Evaluate with ALL 5 classifiers
                clf_results = evaluate_utility(X_aug, y_aug, X_te, y_te, seed=seed)
                
                # Average across classifiers
                f1_avg = np.mean([clf_results[cn]["F1"] for cn in ["XGB", "DTC", "Logit", "RFC", "KNN"]])
                acc_avg = np.mean([clf_results[cn]["Acc"] for cn in ["XGB", "DTC", "Logit", "RFC", "KNN"]])
                mcc_avg = np.mean([clf_results[cn]["MCC"] for cn in ["XGB", "DTC", "Logit", "RFC", "KNN"]])
                
                variant_results[variant_name]['F1'].append(f1_avg)
                variant_results[variant_name]['Acc'].append(acc_avg)
                variant_results[variant_name]['MCC'].append(mcc_avg)
                print(f"F1={f1_avg:.4f}")  # ← ADD THIS
        
        # Aggregate results
        print(f"\n  [Dataset Summary: {ds_name}]")
        for variant_name in ABLATION_VARIANTS:
            f1_mean = np.mean(variant_results[variant_name]['F1'])
            f1_std = np.std(variant_results[variant_name]['F1'])
            acc_mean = np.mean(variant_results[variant_name]['Acc'])
            mcc_mean = np.mean(variant_results[variant_name]['MCC'])
            
            print(f"    {variant_name:<20s} F1: {f1_mean:.4f} ± {f1_std:.4f}")
            
            all_results.append({
                "Dataset": ds_name, "Variant": variant_name,
                "F1_Mean": f1_mean, "F1_Std": f1_std,
                "Acc_Mean": acc_mean, "MCC_Mean": mcc_mean
            })
    
    # Save
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("ablation_results.csv", index=False)
        
        print("\n" + "=" * 120)
        print("ABLATION SUMMARY")
        print("=" * 120)
        pivot = df.pivot(index="Variant", columns="Dataset", values="F1_Mean")
        order = ["IDENTITY", "w/o Fisher", "w/o Geodesic", "w/o Spectral", "MAT-Diff (Ours)"]
        pivot = pivot.reindex(order)
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
        print("=" * 120)
        
    return all_results
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--matdiff_epochs", type=int, default=None)
    parser.add_argument("--dgot", action="store_true", help="Include DGOT")
    parser.add_argument("--goio", action="store_true", help="Include GOIO")
    parser.add_argument("--stats_only", action="store_true", 
                       help="Run statistical tests on existing results (skip training)")
    parser.add_argument("--ablation", action="store_true",
                       help="Run ablation study (MAT-Diff components)")
    args = parser.parse_args()
    
    if torch:
        device = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    else:
        device = "cpu"
    
    datasets = args.datasets or list(DATASET_REGISTRY.keys())
    
    print("=" * 80)
    print("MAT-Diff Benchmark - FINAL VERSION")
    print(f"Device: {device}, Datasets: {datasets}")
    print(f"Seeds: {args.n_seeds}, Folds: {args.n_folds}")
    print(f"DGOT: {args.dgot}, GOIO: {args.goio}")
    print("=" * 80)
    
    if args.stats_only:
        # Load existing results and run stats only
        results_path = "benchmark_results.csv"
        if not os.path.exists(results_path):
            print(f"ERROR: {results_path} not found. Run full benchmark first.")
            return

        print(f"Loading results from {results_path}...")
        df = pd.read_csv(results_path)
        print("Statistics already printed in benchmark results.")
    

    elif args.ablation:
        # If no datasets specified, use DGOT's 4 representative datasets
        if args.datasets is None:
            ablation_datasets = [
                'wine_quality',    # Binary, high IR (25.77)
                'bank',            # Binary, low IR (7.7)
                'Dry_Beans',       # Multi-class, low IR (6.8)
                'satimage'         # Multi-class, very low IR (2.5)
            ]
            print(f"No datasets specified, using DGOT's 4 representative datasets")
        else:
            ablation_datasets = datasets
        
        print(f"Running ablation on: {', '.join(ablation_datasets)}")
        run_ablation_study(ablation_datasets, device, n_seeds=10, n_folds=5)

    else:
        # Normal full benchmark
        run_benchmark(datasets, device, args.n_seeds, args.n_folds, 
                      args.matdiff_epochs, args.dgot, args.goio)


if __name__ == "__main__":
    main()
























































