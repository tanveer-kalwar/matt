"""
Sensitivity analysis for overgeneration factor and hardness threshold.
"""
import numpy as np
import time
from scipy import stats as scipy_stats

from ..data_loader import prepare_data
from ..generators.tabddpm_aug import find_hard_samples, tabddpm_aug_ensemble_generator, dcr_filtering
from ..evaluation.utility import evaluate_comprehensive

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

def run_sensitivity_overgeneration(df, device, main_config):
    """
    Test overgeneration factors (1.0, 1.5, 1.8, 3.0) with 5 seeds.

    Args:
        df: Original input DataFrame used to prepare the dataset.
        device: Compute device to use ('cpu' or 'cuda').
        main_config: Dict of model and training hyperparameters.

    Returns:
        dict: Mapping from overgeneration factor to a list of result dicts
            (utility, fidelity, privacy, and timing metrics).
    """
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: OVERGENERATION FACTORS")
    print("="*80)
    
    factors = [1.0, 1.5, 1.8, 3.0]
    n_seeds = 5 
    results_by_factor = {f: [] for f in factors}
    
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 100
        print(f"\n{'='*60}")
        print(f"Sensitivity Seed {seed_idx+1}/{n_seeds} (seed={seed})")
        print(f"{'='*60}")
        
        data = prepare_data(df, seed)
        
        if data['n_needed'] <= 0:
            continue
        
        for factor in factors:
            print(f"\n  Testing factor: {factor}×")
            try:
                start_t = time.time()
                
                X_minority_hard, X_minority_easy = find_hard_samples(
                    data['X_train_norm'], data['y_train'], 
                    data['X_minority'], data['minority_class'], seed
                )
                
                n_needed = data['n_needed']
                n_smote = int(n_needed * 0.5)
                n_tabddpm = n_needed - n_smote
                
                # SMOTE generation
                X_smote = np.array([]).reshape(0, data['X_minority'].shape[1])
                if SMOTE_AVAILABLE and n_smote > 0 and len(X_minority_easy) > 1:
                    try:
                        majority_class = 0 if data['minority_class'] == 1 else 1
                        X_majority = data['X_train_norm'][data['y_train'] == majority_class]
                        X_temp = np.vstack([X_minority_easy, X_majority[:min(len(X_majority), len(X_minority_easy)*2)]])
                        y_temp = np.hstack([
                            np.full(len(X_minority_easy), data['minority_class']),
                            np.full(min(len(X_majority), len(X_minority_easy)*2), majority_class)
                        ])
                        smote = SMOTE(
                            sampling_strategy={data['minority_class']: len(X_minority_easy) + n_smote},
                            k_neighbors=min(5, len(X_minority_easy) - 1),
                            random_state=seed
                        )
                        X_resampled, y_resampled = smote.fit_resample(X_temp, y_temp)
                        X_smote = X_resampled[len(X_temp):len(X_temp) + n_smote]
                    except:
                        indices = np.random.choice(len(X_minority_easy), n_smote, replace=True)
                        X_smote = X_minority_easy[indices]
                
                # Ensemble with specific factor
                X_pool = tabddpm_aug_ensemble_generator(
                    X_minority_hard, n_tabddpm, main_config, seed, device,
                    n_models=3, overgen_factor=factor
                )
                
                X_tabddpm_final = dcr_filtering(X_pool, data['X_minority'], n_tabddpm, seed=seed)
                
                # Combine
                if len(X_smote) > 0 and len(X_tabddpm_final) > 0:
                    X_final = np.vstack([X_smote, X_tabddpm_final])
                elif len(X_smote) > 0:
                    X_final = X_smote
                else:
                    X_final = X_tabddpm_final
                
                # Ensure exact count
                if len(X_final) > n_needed:
                    indices = np.random.permutation(len(X_final))[:n_needed]
                    X_final = X_final[indices]
                elif len(X_final) < n_needed and len(X_final) > 0:
                    additional = n_needed - len(X_final)
                    indices = np.random.choice(len(X_final), size=additional, replace=True)
                    X_final = np.vstack([X_final, X_final[indices]])
                
                elapsed = time.time() - start_t
                
                r = evaluate_comprehensive(
                    data['X_train_norm'], data['y_train'],
                    data['X_test_norm'], data['y_test'],
                    X_final, f'Factor_{factor}', seed
                )
                r['time'] = elapsed
                r['factor'] = factor
                results_by_factor[factor].append(r)
                
                print(f"    F1={r['f1']:.4f}, MIA={r['mia_auc']:.4f}, Time={elapsed:.1f}s")
                
            except Exception as e:
                print(f"    FAILED: {str(e)}")
    
    # Summary table
    print("\n" + "="*100)
    print("OVERGENERATION SENSITIVITY SUMMARY (5 Seeds)")
    print("="*100)
    print(f"\n{'Factor':<10} {'F1':<20} {'AUC':<20} {'MIA':<20} {'Time (s)':<12}")
    print("-"*90)
    
    for factor in factors:
        if not results_by_factor[factor]:
            print(f"{factor}×{'':<7} {'FAILED':<20}")
            continue
        
        f1_vals = [r['f1'] for r in results_by_factor[factor]]
        auc_vals = [r['auc'] for r in results_by_factor[factor]]
        mia_vals = [r['mia_auc'] for r in results_by_factor[factor]]
        time_vals = [r['time'] for r in results_by_factor[factor]]
        
        print(f"{factor}×{'':<7} {np.mean(f1_vals):.4f}±{np.std(f1_vals):.4f}  "
              f"{np.mean(auc_vals):.4f}±{np.std(auc_vals):.4f}  "
              f"{np.mean(mia_vals):.4f}±{np.std(mia_vals):.4f}  {np.mean(time_vals):.1f}")
    
    # Statistical comparison
    print("\n" + "="*100)
    print("STATISTICAL COMPARISON (vs. Default 1.8×)")
    print("="*100)
    
    baseline_f1 = [r['f1'] for r in results_by_factor[1.8]]
    
    print(f"\n{'Factor':<10} {'F1 t-stat':<15} {'p-value':<15} {'Conclusion':<30}")
    print("-"*75)
    
    for factor in [1.0, 1.5, 3.0]:
        comp_f1 = [r['f1'] for r in results_by_factor[factor]]
        
        if comp_f1 and baseline_f1:
            t_f1, p_f1 = scipy_stats.ttest_rel(baseline_f1, comp_f1)
            sig = "***" if p_f1 < 0.001 else ("**" if p_f1 < 0.01 else ("*" if p_f1 < 0.05 else "n.s."))
            delta = np.mean(comp_f1) - np.mean(baseline_f1)
            conclusion = f"Δ={delta:+.4f} {sig}"
            
            print(f"{factor}×{'':<7} {t_f1:+10.3f}      {p_f1:>10.4f}      {conclusion:<30}")
    
    return results_by_factor


def run_sensitivity_threshold(df, device, main_config):
    """
    Test hardness thresholds (0.5, 0.6, 0.7, 0.8, 0.9) with 5 seeds.

    Args:
        df: Original input DataFrame used to prepare the dataset.
        device: Compute device to use ('cpu' or 'cuda').
        main_config: Dict of model and training hyperparameters.

    Returns:
        dict: Mapping from hardness threshold to a list of result dicts
            (utility, fidelity, and privacy metrics).
    """
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: HARDNESS THRESHOLD")
    print("="*80)
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    n_seeds = 5
    results_by_threshold = {t: [] for t in thresholds}
    
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 100
        print(f"\n{'='*60}")
        print(f"Sensitivity Seed {seed_idx+1}/{n_seeds} (seed={seed})")
        print(f"{'='*60}")
        
        data = prepare_data(df, seed)
        
        if data['n_needed'] <= 0:
            print("Dataset balanced, skipping.")
            continue
        
        for threshold in thresholds:
            print(f"\n  Testing threshold: τ_hard={threshold}")
            try:
                # Custom split with this threshold
                from sklearn.linear_model import LogisticRegression
                
                clf = LogisticRegression(class_weight='balanced', random_state=seed, max_iter=300, n_jobs=-1)
                clf.fit(data['X_train_norm'], data['y_train'])
                
                y_pred_proba = clf.predict_proba(data['X_minority'])[:, data['minority_class']]
                hard_mask = (y_pred_proba < threshold)
                
                X_minority_hard = data['X_minority'][hard_mask]
                X_minority_easy = data['X_minority'][~hard_mask]
                
                if len(X_minority_hard) < 2:
                    X_minority_hard = data['X_minority'][:max(2, int(len(data['X_minority'])*0.2))]
                    X_minority_easy = data['X_minority'][max(2, int(len(data['X_minority'])*0.2)):]
                if len(X_minority_easy) < 2:
                    X_minority_easy = data['X_minority']
                
                print(f"    Split: {len(X_minority_hard)} hard, {len(X_minority_easy)} easy")
                
                # Standard pipeline
                n_needed = data['n_needed']
                n_smote = int(n_needed * 0.5)
                n_tabddpm = n_needed - n_smote
                
                X_smote = np.array([]).reshape(0, data['X_minority'].shape[1])
                if SMOTE_AVAILABLE and n_smote > 0 and len(X_minority_easy) > 1:
                    try:
                        majority_class = 0 if data['minority_class'] == 1 else 1
                        X_majority = data['X_train_norm'][data['y_train'] == majority_class]
                        X_temp = np.vstack([X_minority_easy, X_majority[:min(len(X_majority), len(X_minority_easy)*2)]])
                        y_temp = np.hstack([
                            np.full(len(X_minority_easy), data['minority_class']),
                            np.full(min(len(X_majority), len(X_minority_easy)*2), majority_class)
                        ])
                        smote = SMOTE(
                            sampling_strategy={data['minority_class']: len(X_minority_easy) + n_smote},
                            k_neighbors=min(5, len(X_minority_easy) - 1),
                            random_state=seed
                        )
                        X_resampled, y_resampled = smote.fit_resample(X_temp, y_temp)
                        X_smote = X_resampled[len(X_temp):len(X_temp) + n_smote]
                    except:
                        indices = np.random.choice(len(X_minority_easy), n_smote, replace=True)
                        X_smote = X_minority_easy[indices]
                
                X_pool = tabddpm_aug_ensemble_generator(
                    X_minority_hard, n_tabddpm, main_config, seed, device,
                    n_models=3, overgen_factor=1.8
                )
                
                X_tabddpm_final = dcr_filtering(X_pool, data['X_minority'], n_tabddpm, seed=seed)
                
                if len(X_smote) > 0 and len(X_tabddpm_final) > 0:
                    X_final = np.vstack([X_smote, X_tabddpm_final])
                elif len(X_smote) > 0:
                    X_final = X_smote
                else:
                    X_final = X_tabddpm_final
                
                if len(X_final) > n_needed:
                    indices = np.random.permutation(len(X_final))[:n_needed]
                    X_final = X_final[indices]
                elif len(X_final) < n_needed and len(X_final) > 0:
                    additional = n_needed - len(X_final)
                    indices = np.random.choice(len(X_final), size=additional, replace=True)
                    X_final = np.vstack([X_final, X_final[indices]])
                
                r = evaluate_comprehensive(
                    data['X_train_norm'], data['y_train'],
                    data['X_test_norm'], data['y_test'],
                    X_final, f'Threshold_{threshold}', seed
                )
                r['threshold'] = threshold
                results_by_threshold[threshold].append(r)
                
                print(f"    F1={r['f1']:.4f}, MIA={r['mia_auc']:.4f}")
                
            except Exception as e:
                print(f"    FAILED: {str(e)}")
    
    # Summary
    print("\n" + "="*100)
    print("HARDNESS THRESHOLD SENSITIVITY SUMMARY (5 Seeds)")
    print("="*100)
    print(f"\n{'Threshold':<15} {'F1':<20} {'AUC':<20} {'MIA':<20}")
    print("-"*75)
    
    for threshold in thresholds:
        if not results_by_threshold[threshold]:
            print(f"{threshold:<15} {'FAILED':<20}")
            continue
        
        f1_vals = [r['f1'] for r in results_by_threshold[threshold]]
        auc_vals = [r['auc'] for r in results_by_threshold[threshold]]
        mia_vals = [r['mia_auc'] for r in results_by_threshold[threshold]]
        
        print(f"{threshold:<15} {np.mean(f1_vals):.4f}±{np.std(f1_vals):.4f}  "
              f"{np.mean(auc_vals):.4f}±{np.std(auc_vals):.4f}  "
              f"{np.mean(mia_vals):.4f}±{np.std(mia_vals):.4f}")
    
    # Statistical comparison
    print("\n" + "="*100)
    print("STATISTICAL COMPARISON (vs. Default τ=0.7)")
    print("="*100)
    
    baseline_f1 = [r['f1'] for r in results_by_threshold[0.7]]
    
    print(f"\n{'Threshold':<15} {'F1 t-stat':<15} {'p-value':<15} {'Conclusion':<30}")
    print("-"*75)
    
    for threshold in [0.5, 0.6, 0.8, 0.9]:
        comp_f1 = [r['f1'] for r in results_by_threshold[threshold]]
        
        if comp_f1 and baseline_f1:
            t_f1, p_f1 = scipy_stats.ttest_rel(baseline_f1, comp_f1)
            delta = np.mean(comp_f1) - np.mean(baseline_f1)
            
            if abs(delta) < 0.01:
                conclusion = "Robust (negligible Δ)"
            else:
                sig = "***" if p_f1 < 0.001 else ("**" if p_f1 < 0.01 else ("*" if p_f1 < 0.05 else "n.s."))
                conclusion = f"Δ={delta:+.4f} {sig}"
            
            print(f"{threshold:<15} {t_f1:+10.3f}      {p_f1:>10.4f}      {conclusion:<30}")
    
    return results_by_threshold
