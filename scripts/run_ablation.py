#!/usr/bin/env python3
"""
Run ablation study to compare adaptive vs fixed strategies.

Usage:
    python scripts/run_ablation.py --dataset pima
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
from scipy import stats as scipy_stats
sys.path.insert(0, str(Path(__file__).parent.parent))

from TabDDPM_Aug.data_loader import load_dataset, prepare_data
from TabDDPM_Aug.config import get_config
from TabDDPM_Aug.experiments.ablation import generate_ablation_variant
from TabDDPM_Aug.evaluation.utility import evaluate_comprehensive


def run_ablation_studies(df, device, main_config, dataset_name):
    """Run multi-seed ablation comparing adaptive vs fixed strategies.

    Args:
        df (pandas.DataFrame): Preprocessed dataset containing features and a binary 'target' column.
        device (str or torch.device): Computation device, e.g. "cpu" or "cuda".
        main_config (dict): Dataset-specific hyperparameters, including 'n_seeds' and TabDDPM settings.
        dataset_name (str): Name of the dataset used in logs and summaries.

    Returns:
        None: Prints per-seed results, aggregated metrics, and paired
            t-test statistics to stdout.
    """
    print("\n" + "="*80)
    print("RUNNING ADAPTIVE ABLATION STUDIES")
    print("="*80)
    
    n_seeds = main_config['n_seeds']
    
    # Define the variants
    ablation_configs = {
        'Adaptive_AHA_DCR': {
            'name': '1. Our Adaptive Model (AHA-DCR)',
            'split_strategy': 'adaptive',
            'filter_mode': 'dcr'
        },
        'Fixed_DBHA_Split': {
            'name': '2. Fixed "Pima" Strategy (DBHA-DCR)',
            'split_strategy': 'force_dbha',
            'filter_mode': 'dcr'
        },
        'Fixed_FDHA_Hybrid': {
            'name': '3. Fixed "Adult" Strategy (FDHA-DCR)',
            'split_strategy': 'force_fdha',
            'filter_mode': 'dcr'
        },
        'Ablate_DCR_Filter': {
            'name': '4. Ablate Filter (Adaptive + No Filter)',
            'split_strategy': 'adaptive',
            'filter_mode': 'none'
        }
    }
    
    results = {name: [] for name in ablation_configs.keys()}

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 100
        print(f"\n{'='*80}")
        print(f"Ablation Seed {seed_idx+1}/{n_seeds} (seed={seed})")
        print(f"{'='*80}")
        
        data = prepare_data(df, seed)
        
        if data['n_needed'] <= 0:
            print("Dataset is balanced, skipping ablation seed.")
            continue
        
        for name, config in ablation_configs.items():
            try:
                X_synthetic = generate_ablation_variant(
                    config, data, main_config, seed, device
                )
                
                if X_synthetic is not None and len(X_synthetic) > 0:
                    r = evaluate_comprehensive(
                        data['X_train_norm'], data['y_train'],
                        data['X_test_norm'], data['y_test'],
                        X_synthetic, name, seed
                    )
                    results[name].append(r)
                    print(f"  {name}: F1={r['f1']:.4f}, KS={r['ks_statistic']:.4f}, MIA={r['mia_auc']:.4f}")
                else:
                    print(f"  {name}: Failed to generate samples")
            except Exception as e:
                print(f"  {name}: FAILED with error: {e}")
                import traceback
                traceback.print_exc()

    # Print Final Ablation Summary
    print("\n" + "="*100)
    print(f"FINAL ABLATION RESULTS ({dataset_name}) - (Mean ± Std. Dev. over {n_seeds} Seeds)")
    print("="*100)
    print(f"\n{'Method':<35} {'F1 ↑':<18} {'KS ↓':<18} {'MIA ↓':<18}")
    print("-"*90)

    baseline_results = results['Adaptive_AHA_DCR']

    for name, res_list in results.items():
        if not res_list:
            print(f"{name:<35} {'FAILED':<18} {'FAILED':<18} {'FAILED':<18}")
            continue
            
        f1_vals = [r['f1'] for r in res_list]
        ks_vals = [r['ks_statistic'] for r in res_list]
        mia_vals = [r['mia_auc'] for r in res_list]
        
        f1_str = f"{np.mean(f1_vals):.4f}±{np.std(f1_vals):.4f}"
        ks_str = f"{np.mean(ks_vals):.4f}±{np.std(ks_vals):.4f}"
        mia_str = f"{np.mean(mia_vals):.4f}±{np.std(mia_vals):.4f}"
        
        print(f"{ablation_configs[name]['name']:<35} {f1_str:<18} {ks_str:<18} {mia_str:<18}")

    print("\n" + "="*100)
    print("STATISTICAL SIGNIFICANCE (vs. Our Adaptive Model)")
    print("="*100)
    print(f"\n{'Ablated Model':<35} {'F1 (t-stat, p-val)':<25} {'MIA (t-stat, p-val)':<25}")
    print("-"*85)

    
    def paired_t_test(vals1, vals2):
        if len(vals1) < 2 or len(vals2) < 2: return np.nan, 1.0
        min_len = min(len(vals1), len(vals2))
        return scipy_stats.ttest_rel(vals1[:min_len], vals2[:min_len])

    if baseline_results:
        baseline_f1 = [r['f1'] for r in baseline_results]
        baseline_mia = [r['mia_auc'] for r in baseline_results]

        for name, res_list in results.items():
            if name == 'Adaptive_AHA_DCR' or not res_list:
                continue
            
            f1_vals = [r['f1'] for r in res_list]
            mia_vals = [r['mia_auc'] for r in res_list]
            
            t_f1, p_f1 = paired_t_test(baseline_f1, f1_vals) 
            t_mia, p_mia = paired_t_test(baseline_mia, mia_vals)

            f1_str = f"t={t_f1: 5.2f}, p={p_f1:.3f}"
            mia_str = f"t={t_mia: 5.2f}, p={p_mia:.3f}"
            
            print(f"{ablation_configs[name]['name']:<35} {f1_str:<25} {mia_str:<25}")

def main():
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['adult', 'pima', 'credit'])
    args = parser.parse_args()
    
    print("="*80)
    print("TabDDPM-Aug: Ablation Study")
    print(f"Dataset: {args.dataset}")
    print("="*80)
    
    df = load_dataset(args.dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    main_config = get_config(args.dataset)
    run_ablation_studies(df, device, main_config, args.dataset)
    
    print("\n" + "="*80)
    print("ABLATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
