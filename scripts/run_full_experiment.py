#!/usr/bin/env python3
"""
Full 9-method comparison experiment for TabDDPM-Aug.

Compares:
    1. Original (no augmentation)
    2-3. SMOTE, ADASYN
    4-7. Copula, CTGAN, TVAE, CTAB-GAN+
    8. TabDDPM (baseline)
    9. TabDDPM-Aug (proposed)

Usage:
    python scripts/run_full_experiment.py --dataset pima
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import time
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from TabDDPM_Aug.data_loader import load_dataset, prepare_data
from TabDDPM_Aug.config import get_config
from TabDDPM_Aug.generators.smote_baseline import smote_adasyn_generator, SMOTE_AVAILABLE
from TabDDPM_Aug.generators.gan_baselines import copula_generator, ctgan_generator, tvae_generator, ctabgan_plus_generator
from TabDDPM_Aug.generators.tabddpm_baseline import tabddpm_generator_original_baseline
from TabDDPM_Aug.generators.tabddpm_aug import tabddpm_aug_final
from TabDDPM_Aug.evaluation.utility import evaluate_simple, evaluate_comprehensive

if SMOTE_AVAILABLE:
    from imblearn.over_sampling import SMOTE, ADASYN


def run_full_experiment(df, device, main_config, dataset_name):
    """Execute full 9-method multi-seed comparison of all augmentation methods.

    Args:
        df (pandas.DataFrame): Preprocessed dataset containing features and a binary 'target' column.
        device (str or torch.device): Computation device, e.g. "cpu" or "cuda".
        main_config (dict): Dataset-specific hyperparameters, including 'n_seeds' and TabDDPM settings.
        dataset_name (str): Dataset identifier used for logging and summaries.

    Returns:
        None: Prints per-seed metrics, aggregated tables, and paired
            t-test statistics to stdout.
    """
    print("\n" + "="*80)
    print("RUNNING FULL EXPERIMENT")
    print("="*80)

    n_seeds = main_config['n_seeds']
    print(f"Running {n_seeds} seeds for statistical significance\n")
    
    results = {
        'original': [], 'smote': [], 'adasyn': [], 'copula': [],
        'ctgan': [], 'tvae': [], 'ctabganp': [], 'tabddpm': [], 'tabddpm_aug': []
    }
    timing_info = defaultdict(list)
    
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 100
        print(f"\n{'='*80}")
        print(f"Seed {seed_idx+1}/{n_seeds} (seed={seed})")
        print(f"{'='*80}")
        
        data = prepare_data(df, seed)
        
        if data['n_needed'] <= 0:
            print("Dataset is balanced, skipping seed.")
            continue
        
        # ========================================================================
        # Method 1: Original
        # ========================================================================            
        print("\n[1/9] Original (No Augmentation)")
        r = evaluate_simple(data['X_train_norm'], data['y_train'], data['X_test_norm'], data['y_test'], 'Original', seed)
        results['original'].append(r)
        print(f"  F1={r['f1']:.4f}, AUC={r['auc']:.4f}, AUPRC={r['auprc']:.4f}")
        
        # ========================================================================
        # Method 2: SMOTE
        # ========================================================================
        print("\n[2/9] SMOTE")
        start_t = time.time()
        X_aug_smote, y_aug_smote = smote_adasyn_generator(data['X_train_norm'], data['y_train'], SMOTE, seed)
        elapsed = time.time() - start_t
        timing_info['smote'].append(elapsed)
        r = evaluate_simple(X_aug_smote, y_aug_smote, data['X_test_norm'], data['y_test'], 'SMOTE', seed)
        results['smote'].append(r)
        print(f"  F1={r['f1']:.4f}, AUC={r['auc']:.4f}, AUPRC={r['auprc']:.4f}, Time={elapsed:.2f}s")
        
        # ========================================================================
        # Method 3: ADASYN
        # ========================================================================
        print("\n[3/9] ADASYN")
        start_t = time.time()
        X_aug_adasyn, y_aug_adasyn = smote_adasyn_generator(data['X_train_norm'], data['y_train'], ADASYN, seed)
        elapsed = time.time() - start_t
        timing_info['adasyn'].append(elapsed)
        r = evaluate_simple(X_aug_adasyn, y_aug_adasyn, data['X_test_norm'], data['y_test'], 'ADASYN', seed)
        results['adasyn'].append(r)
        print(f"  F1={r['f1']:.4f}, AUC={r['auc']:.4f}, AUPRC={r['auprc']:.4f}, Time={elapsed:.2f}s")
        
        # ========================================================================
        # Method 4: Copula
        # ========================================================================         
        print("\n[4/9] GaussianCopula")
        try:
            start_t = time.time()
            X_copula = copula_generator(data['X_minority_df'], data['n_needed'], data['categorical_cols'], 
                                        data['scaler'], data['label_encoders'], data['numeric_cols'], seed)
            elapsed = time.time() - start_t
            timing_info['copula'].append(elapsed)
            r = evaluate_comprehensive(data['X_train_norm'], data['y_train'], data['X_test_norm'], 
                                       data['y_test'], X_copula, 'Copula', seed)
            results['copula'].append(r)
            print(f"  F1={r['f1']:.4f}, AUC={r['auc']:.4f}, AUPRC={r['auprc']:.4f}, Time={elapsed:.2f}s")
            print(f"  KS={r['ks_statistic']:.4f}, MMD={r['mmd']:.4f}, DCR={r['mean_dcr']:.2f}, MIA={r['mia_auc']:.4f}")
        except Exception as e:
            print(f"  SKIPPED: {str(e)}")
            
        # ========================================================================
        # Method 5: CTGAN
        # ========================================================================         
        print("\n[5/9] CTGAN")
        try:
            start_t = time.time()
            X_ctgan = ctgan_generator(data['X_minority_df'], data['X_minority'], data['n_needed'], 
                                      data['categorical_cols'], data['scaler'], data['label_encoders'], 
                                      data['numeric_cols'], seed)
            elapsed = time.time() - start_t
            timing_info['ctgan'].append(elapsed)
            r = evaluate_comprehensive(data['X_train_norm'], data['y_train'], data['X_test_norm'], 
                                       data['y_test'], X_ctgan, 'CTGAN', seed)
            results['ctgan'].append(r)
            print(f"  F1={r['f1']:.4f}, AUC={r['auc']:.4f}, AUPRC={r['auprc']:.4f}, Time={elapsed:.2f}s")
            print(f"  KS={r['ks_statistic']:.4f}, MMD={r['mmd']:.4f}, DCR={r['mean_dcr']:.2f}, MIA={r['mia_auc']:.4f}")
        except Exception as e:
            print(f"  SKIPPED: {str(e)}")
            
        # ========================================================================
        # Method 6: TVAE
        # ========================================================================         
        print("\n[6/9] TVAE")
        try:
            start_t = time.time()
            X_tvae = tvae_generator(data['X_minority_df'], data['X_minority'], data['n_needed'], 
                                    data['categorical_cols'], data['scaler'], data['label_encoders'], 
                                    data['numeric_cols'], seed)
            elapsed = time.time() - start_t
            timing_info['tvae'].append(elapsed)
            r = evaluate_comprehensive(data['X_train_norm'], data['y_train'], data['X_test_norm'], 
                                       data['y_test'], X_tvae, 'TVAE', seed)
            results['tvae'].append(r)
            print(f"  F1={r['f1']:.4f}, AUC={r['auc']:.4f}, AUPRC={r['auprc']:.4f}, Time={elapsed:.2f}s")
            print(f"  KS={r['ks_statistic']:.4f}, MMD={r['mmd']:.4f}, DCR={r['mean_dcr']:.2f}, MIA={r['mia_auc']:.4f}")
        except Exception as e:
            print(f"  SKIPPED: {str(e)}")
            
        # ========================================================================
        # Method 7: CTAB-GAN-Plus
        # ========================================================================         
        print("\n[7/9] CTAB-GAN-Plus")
        try:
            start_t = time.time()
            X_ctabganp = ctabgan_plus_generator(data['X_minority_df'], data['X_minority'], data['n_needed'], 
                                                data['categorical_cols'], data['scaler'], data['label_encoders'], 
                                                data['numeric_cols'], data['minority_class'], seed)
            elapsed = time.time() - start_t
            timing_info['ctabganp'].append(elapsed)
            r = evaluate_comprehensive(data['X_train_norm'], data['y_train'], data['X_test_norm'], 
                                       data['y_test'], X_ctabganp, 'CTABGAN-Plus', seed)
            results['ctabganp'].append(r)
            print(f"  F1={r['f1']:.4f}, AUC={r['auc']:.4f}, AUPRC={r['auprc']:.4f}, Time={elapsed:.2f}s")
            print(f"  KS={r['ks_statistic']:.4f}, MMD={r['mmd']:.4f}, DCR={r['mean_dcr']:.2f}, MIA={r['mia_auc']:.4f}")
        except Exception as e:
            print(f"  SKIPPED: {str(e)}")
            
        # ========================================================================
        # Method 8: TabDDPM Baseline
        # ========================================================================         
        print("\n[8/9] TabDDPM (Original Baseline)")
        try:
            start_t = time.time()
            X_tabddpm = tabddpm_generator_original_baseline(
                data['X_minority'], data['n_needed'], main_config, seed, device
            )
            elapsed = time.time() - start_t
            timing_info['tabddpm'].append(elapsed)
            r = evaluate_comprehensive(data['X_train_norm'], data['y_train'], 
                                       data['X_test_norm'], data['y_test'], 
                                       X_tabddpm, 'TabDDPM', seed)
            results['tabddpm'].append(r)
            print(f"  F1={r['f1']:.4f}, AUC={r['auc']:.4f}, AUPRC={r['auprc']:.4f}, Time={elapsed:.2f}s")
            print(f"  KS={r['ks_statistic']:.4f}, MMD={r['mmd']:.4f}, DCR={r['mean_dcr']:.2f}, MIA={r['mia_auc']:.4f}")
        except Exception as e:
            print(f"  SKIPPED: {str(e)}")
            
        # ========================================================================
        # Method 9: TabDDPM-Aug
        # ========================================================================         
        print("\n[9/9] TabDDPM-Aug (Adaptive-DBHA-DCR)")
        try:
            start_t = time.time()
            X_tabddpm_aug = tabddpm_aug_final(
                data['X_train_norm'], data['y_train'], main_config, seed, device
            )
            elapsed = time.time() - start_t
            timing_info['tabddpm_aug'].append(elapsed)
            
            if X_tabddpm_aug is not None and len(X_tabddpm_aug) > 0:
                r = evaluate_comprehensive(data['X_train_norm'], data['y_train'], 
                                           data['X_test_norm'], data['y_test'], 
                                           X_tabddpm_aug, 'TabDDPM-Aug', seed)
                results['tabddpm_aug'].append(r)
                print(f"  F1={r['f1']:.4f}, AUC={r['auc']:.4f}, AUPRC={r['auprc']:.4f}, Time={elapsed:.2f}s")
                print(f"  KS={r['ks_statistic']:.4f}, MMD={r['mmd']:.4f}, DCR={r['mean_dcr']:.2f}, MIA={r['mia_auc']:.4f}")
            else:
                print("  TabDDPM-Aug failed to generate samples")
        except Exception as e:
            print(f"  FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print Final Full Experiment Summary
    print("\n" + "="*100)
    print("FINAL RESULTS - Classification Performance")
    print("="*100)
    print(f"\n{'Method':<20} {'F1':<18} {'AUC':<18} {'AUPRC':<18} {'Time (s)':<12}")
    print("-"*100)
    
    for method in ['original', 'smote', 'adasyn', 'copula', 'ctgan', 'tvae', 'ctabganp', 'tabddpm', 'tabddpm_aug']:
        f1_vals = [r.get('f1', np.nan) for r in results[method]]
        auc_vals = [r.get('auc', np.nan) for r in results[method]]
        auprc_vals = [r.get('auprc', np.nan) for r in results[method]]
        if not f1_vals:
            continue
        time_str = f"{np.mean(timing_info[method]):.1f}" if method in timing_info else "N/A"
        print(f"{method:<20} {np.nanmean(f1_vals):.4f}±{np.nanstd(f1_vals):.4f}  "
              f"{np.nanmean(auc_vals):.4f}±{np.nanstd(auc_vals):.4f}  "
              f"{np.nanmean(auprc_vals):.4f}±{np.nanstd(auprc_vals):.4f}  "
              f"{time_str:<12}")
    
    print("\n" + "="*100)
    print("FINAL RESULTS - Statistical Fidelity & Privacy")
    print("="*100)
    print(f"\n{'Method':<20} {'KS ↓':<10} {'MMD ↓':<10} {'JS ↓':<10} {'WD ↓':<10} {'Corr ↓':<10} {'DCR':<10} {'MIA ↓':<10}")
    print("-"*120)
    
    for method in ['copula', 'ctgan', 'tvae', 'ctabganp', 'tabddpm', 'tabddpm_aug']:
        if method not in results or not results[method] or 'ks_statistic' not in results[method][0]:
            continue
        ks_vals = [r.get('ks_statistic', np.nan) for r in results[method]]
        mmd_vals = [r.get('mmd', np.nan) for r in results[method]]
        js_vals = [r.get('js_divergence', np.nan) for r in results[method]]
        wd_vals = [r.get('wasserstein', np.nan) for r in results[method]]
        corr_vals = [r.get('correlation_l2', np.nan) for r in results[method]]
        dcr_vals = [r.get('mean_dcr', np.nan) for r in results[method]]
        mia_vals = [r.get('mia_auc', np.nan) for r in results[method]]
        
        print(f"{method:<20} {np.nanmean(ks_vals):.4f}   {np.nanmean(mmd_vals):.4f}   "
              f"{np.nanmean(js_vals):.4f}   {np.nanmean(wd_vals):.4f}   "
              f"{np.nanmean(corr_vals):.2f}      {np.nanmean(dcr_vals):.2f}      "
              f"{np.nanmean(mia_vals):.4f}")
    
    # Statistical Significance
    print("\n" + "="*100)
    print("STATISTICAL SIGNIFICANCE (Paired t-tests)")
    print("="*100)
    
    from scipy import stats as scipy_stats
    
    def paired_t_test(results1, results2, metric='f1'):
        vals1 = [r[metric] for r in results1 if metric in r]
        vals2 = [r[metric] for r in results2 if metric in r]
        
        if len(vals1) < 2 or len(vals2) < 2:
            return np.nan, 1.0
        
        min_len = min(len(vals1), len(vals2))
        vals1, vals2 = vals1[:min_len], vals2[:min_len]
        
        t_stat, p_value = scipy_stats.ttest_rel(vals1, vals2)
        return t_stat, p_value
    
    if results['tabddpm_aug']:
        print(f"\nTabDDPM-Aug (Adaptive-DCR) vs Baselines (F1 Score) - {dataset_name}:")
        print(f"{'Method':<20} {'t-statistic':<15} {'p-value':<15} {'Significant?':<15}")
        print("-"*70)
        
        for method in ['smote', 'adasyn', 'copula', 'ctgan', 'tvae', 'ctabganp', 'tabddpm']:
            if not results[method]:
                continue
            
            t_stat, p_val = paired_t_test(results['tabddpm_aug'], results[method], 'f1')
            
            if t_stat > 0:
                if p_val < 0.001:
                    sig_str = "p<0.001 ***"
                elif p_val < 0.01:
                    sig_str = "p<0.01 **"
                elif p_val < 0.05:
                    sig_str = "p<0.05 *"
                else:
                    sig_str = "(Better, n.s.)"
            else:
                if p_val < 0.05:
                     sig_str = "(Worse, sig.)"
                else:
                     sig_str = "(Worse, n.s.)"
            
            print(f"{method:<20} {t_stat:>10.3f}       {p_val:>10.4f}       {sig_str:<15}")
    

def main():
    parser = argparse.ArgumentParser(description='Run full 8-method comparison')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['adult', 'pima', 'credit'],
                        help='Dataset to use for experiments')
    args = parser.parse_args()
    
    print("="*80)
    print("TabDDPM-Aug: Full Experiment")
    print(f"Dataset: {args.dataset}")
    print("="*80)
    
    df = load_dataset(args.dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    main_config = get_config(args.dataset)
    
    run_full_experiment(df, device, main_config, args.dataset)
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPLETE")
    print("="*100)


if __name__ == '__main__':
    main()
