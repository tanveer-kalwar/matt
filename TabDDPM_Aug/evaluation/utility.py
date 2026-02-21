"""
Classification utility metrics for augmented datasets.
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats
import os
from .fidelity import compute_ks_statistics, compute_mmd, compute_correlation_distance, calculate_js_divergence, compute_wasserstein_distance, compute_dcr_statistics
from .privacy import membership_inference_attack


def evaluate_comprehensive(X_train, y_train, X_test, y_test, X_synthetic, method_name, seed=42):
    """
    Comprehensive evaluation: utility + fidelity + privacy.
    NOW SUPPORTS MULTI-CLASS.
    """
    results = {}
    
    if X_synthetic is None or len(X_synthetic) == 0:
        print("    No synthetic data to evaluate.")
        return {
            'f1_macro': 0.0, 'f1_weighted': 0.0, 'balanced_accuracy': 0.0,
            'auc': 0.5, 'auprc': 0.0, 
            'ks_statistic': 1.0, 'mmd': 1.0, 'js_divergence': 1.0, 
            'wasserstein': 1.0, 'correlation_l2': 10.0, 'mean_dcr': 10.0, 
            'mia_auc': 0.5, 'sdqs': 0.0
        }
    
    # Determine if binary or multi-class
    n_classes = len(np.unique(y_train))
    
    # Get minority class samples for fidelity comparison
    unique, counts = np.unique(y_train, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    X_minority = X_train[y_train == minority_class]
    
    if len(X_minority) == 0:
        X_minority = X_train  # Fallback
    
    # Create augmented dataset
    X_aug = np.vstack([X_train, X_synthetic])
    y_aug = np.hstack([y_train, np.full(len(X_synthetic), minority_class)])
    
    # Classification utility
    catboost_train_dir = '/content/catboost_info/' if os.path.exists('/content/drive') else None
    
    # Multi-class compatible classifiers
    CLASSIFIERS = [
        RandomForestClassifier(n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1),
        XGBClassifier(n_estimators=150, max_depth=8, use_label_encoder=False, 
                     eval_metric='logloss', random_state=seed, n_jobs=-1, 
                     tree_method='hist'),
        LogisticRegression(max_iter=500, random_state=seed)
    ]
    
    # Add CatBoost only if multi-class compatible
    if n_classes == 2:
        unique_aug, counts_aug = np.unique(y_aug, return_counts=True)
        scale_pos = counts_aug[0] / counts_aug[1] if len(counts_aug) == 2 and counts_aug[1] > 0 else 1.0
        CLASSIFIERS.insert(0, CatBoostClassifier(
            iterations=150, learning_rate=0.1, depth=8, 
            scale_pos_weight=min(scale_pos, 40), 
            random_seed=seed, verbose=False, thread_count=-1, 
            train_dir=catboost_train_dir
        ))
    else:
        CLASSIFIERS.insert(0, CatBoostClassifier(
            iterations=150, learning_rate=0.1, depth=8, 
            random_seed=seed, verbose=False, thread_count=-1, 
            train_dir=catboost_train_dir, loss_function='MultiClass'
        ))
    
    f1_macro_scores, f1_weighted_scores, balanced_acc_scores = [], [], []
    auc_scores, auprc_scores = [], []
    
    for clf in CLASSIFIERS:
        try:
            clf.fit(X_aug, y_aug)
            y_pred = clf.predict(X_test)
            
            # Multi-class metrics
            f1_macro_scores.append(f1_score(y_test, y_pred, average='macro'))
            f1_weighted_scores.append(f1_score(y_test, y_pred, average='weighted'))
            balanced_acc_scores.append(balanced_accuracy_score(y_test, y_pred))
            
            # AUC and AUPRC
            if hasattr(clf, 'predict_proba'):
                y_proba = clf.predict_proba(X_test)
                
                if n_classes == 2:
                    auc_scores.append(roc_auc_score(y_test, y_proba[:, 1]))
                    auprc_scores.append(average_precision_score(y_test, y_proba[:, 1]))
                else:
                    # Multi-class one-vs-rest
                    auc_scores.append(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))
                    auprc_scores.append(average_precision_score(y_test, y_proba, average='macro'))
            else:
                # Fallback for classifiers without predict_proba
                auc_scores.append(0.5)
                auprc_scores.append(0.0)
                
        except Exception as e:
            print(f"    Classifier {type(clf).__name__} failed: {e}")
            f1_macro_scores.append(0.0)
            f1_weighted_scores.append(0.0)
            balanced_acc_scores.append(0.0)
            auc_scores.append(0.5)
            auprc_scores.append(0.0)
    
    results['f1_macro'] = float(np.mean(f1_macro_scores))
    results['f1_weighted'] = float(np.mean(f1_weighted_scores))
    results['balanced_accuracy'] = float(np.mean(balanced_acc_scores))
    results['auc'] = float(np.mean(auc_scores))
    results['auprc'] = float(np.mean(auprc_scores))
    
    # Fidelity metrics
    ks_stat, ks_pval = compute_ks_statistics(X_minority, X_synthetic)
    results['ks_statistic'] = float(ks_stat)
    results['ks_pvalue'] = float(ks_pval)
    
    mmd_val = compute_mmd(
        X_minority[:min(1000, len(X_minority))], 
        X_synthetic[:min(1000, len(X_synthetic))], 
        gamma=0.1
    )
    results['mmd'] = float(mmd_val)
    
    try:
        results['js_divergence'] = float(calculate_js_divergence(X_minority, X_synthetic))
    except:
        results['js_divergence'] = float('nan')
    
    try:
        results['wasserstein'] = float(compute_wasserstein_distance(X_minority, X_synthetic))
    except:
        results['wasserstein'] = float('nan')
    
    corr_dist = compute_correlation_distance(X_minority, X_synthetic)
    results['correlation_l2'] = float(corr_dist)
    
    dcr_stats = compute_dcr_statistics(X_minority, X_synthetic)
    results.update(dcr_stats)
    
    mia_auc = membership_inference_attack(X_minority, X_synthetic, X_test)
    results['mia_auc'] = float(mia_auc)
    
    sdqs_results = compute_synthetic_data_quality_score(X_minority, X_synthetic, method_name)
    results.update(sdqs_results)
    
    return results


def evaluate_simple(X_train, y_train, X_test, y_test, method_name, seed):
    """
    IEEE TKDE-standard multi-class utility evaluation.
    
    Metrics:
    - F1-Macro: Unweighted mean of per-class F1 (primary metric for imbalance)
    - F1-Weighted: Sample-weighted F1
    - F1-Micro: Overall accuracy equivalent
    - Balanced Accuracy: Mean of per-class recall
    - G-Mean: Geometric mean of per-class recall (imbalance-specific)
    - AUC: One-vs-rest macro-averaged
    - AUPRC: Average precision macro-averaged
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # Multi-class F1 variants
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # G-Mean calculation (geometric mean of per-class recall)
    cm = confusion_matrix(y_test, y_pred)
    per_class_recall = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
    gmean = np.prod(per_class_recall) ** (1.0 / len(per_class_recall))
    
    # AUC metrics
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
        auprc = average_precision_score(y_test, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        auprc = average_precision_score(y_test, y_proba, average='macro')
    
    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'balanced_accuracy': balanced_acc,
        'gmean': gmean,
        'auc': auc,
        'auprc': auprc
    }


def compute_synthetic_data_quality_score(real_data, synthetic_data, method_name='Unknown'):
    """Composite quality metric combining fidelity, utility, and privacy."""
    if len(real_data) == 0 or len(synthetic_data) == 0:
        return {'sdqs': 0.0, 'fidelity_score': 0.0, 'utility_score': 0.0, 'privacy_score': 0.0}

    # 1. Distributional Fidelity (inverse KS statistic)
    ks_stats = []
    for i in range(real_data.shape[1]):
        ks_stat, _ = stats.ks_2samp(real_data[:, i], synthetic_data[:, i])
        ks_stats.append(ks_stat)
    fidelity_score = 1.0 - np.nanmean(ks_stats)
    
    # 2. Utility Score (TSTR)
    try:
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        y_synthetic = np.ones(len(synthetic_data))
        y_real = np.ones(len(real_data))
        
        clf.fit(synthetic_data, y_synthetic)
        utility_score = clf.score(real_data, y_real)
    except:
        utility_score = 0.5
    
    # 3. Privacy Score (based on DCR)
    dcr_stats = compute_dcr_statistics(real_data, synthetic_data)
    privacy_score = np.clip(dcr_stats['mean_dcr'] / 5.0, 0, 1)
    
    sdqs = 0.3 * fidelity_score + 0.4 * utility_score + 0.3 * privacy_score
    
    print(f"    SDQS for {method_name}: {sdqs:.4f} "
          f"(Fidelity={fidelity_score:.3f}, Utility={utility_score:.3f}, Privacy={privacy_score:.3f})")
    
    return {
        'sdqs': float(sdqs),
        'fidelity_score': float(fidelity_score),
        'utility_score': float(utility_score),
        'privacy_score': float(privacy_score)
    }
