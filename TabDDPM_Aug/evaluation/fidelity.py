"""
Distributional fidelity metrics for synthetic data evaluation.
"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors


def compute_ks_statistics(real_data, synthetic_data):
    """
    Compute Kolmogorov-Smirnov statistics across features.

    Args:
        real_data: Real samples, shape (n_real, n_features).
        synthetic_data: Synthetic samples, shape (n_synth, n_features).

    Returns:
        tuple: Mean KS statistic and mean KS p-value across features.
    """
    ks_stats, p_values = [], []
    if real_data.size == 0 or synthetic_data.size == 0:
        return np.nan, np.nan
    for i in range(real_data.shape[1]):
        ks_stat, p_val = stats.ks_2samp(real_data[:, i], synthetic_data[:, i])
        ks_stats.append(ks_stat)
        p_values.append(p_val)
    return np.mean(ks_stats), np.mean(p_values)

def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy with RBF kernel.

    Args:
        X: First sample set, shape (n_x, n_features).
        Y: Second sample set, shape (n_y, n_features).
        kernel: Kernel type to use ('rbf' supported).
        gamma: RBF kernel bandwidth parameter.

    Returns:
        float: Estimated MMD value between X and Y.
    """
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return np.nan
    if kernel == 'rbf':
        XX = np.sum(X**2, axis=1)[:, None]
        YY = np.sum(Y**2, axis=1)[:, None]
        XY = np.dot(X, Y.T)
        K_XX = np.exp(-gamma * (XX + XX.T - 2 * np.dot(X, X.T)))
        K_YY = np.exp(-gamma * (YY + YY.T - 2 * np.dot(Y, Y.T)))
        K_XY = np.exp(-gamma * (XX + YY.T - 2 * XY))
        mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
        return max(0, mmd)
    return 0.0

def compute_correlation_distance(real_data, synthetic_data):
    """
    Compute Frobenius norm of correlation matrix difference.

    Args:
        real_data: Real samples, shape (n_real, n_features).
        synthetic_data: Synthetic samples, shape (n_synth, n_features).

    Returns:
        float: Frobenius norm between real and synthetic correlation matrices.
    """
    if real_data.size == 0 or synthetic_data.size == 0:
        return np.nan
    if np.isnan(real_data).any() or np.isnan(synthetic_data).any():
        return np.nan
    try:
        real_corr = np.corrcoef(real_data.T)
        synth_corr = np.corrcoef(synthetic_data.T)
        if np.isnan(real_corr).any() or np.isnan(synth_corr).any():
             return np.nan
        return np.linalg.norm(real_corr - synth_corr, 'fro')
    except Exception:
        return np.nan

def compute_dcr_statistics(real_data, synthetic_data):
    """
    Compute Distance-to-Closest-Real (DCR) statistics.

    Args:
        real_data: Real samples used as reference set.
        synthetic_data: Synthetic samples whose DCR is measured.

    Returns:
        dict: DCR statistics with keys 'mean_dcr', 'median_dcr',
            'min_dcr', and 'pct_above_4'.
    """
    if len(real_data) == 0 or len(synthetic_data) == 0:
        return {'mean_dcr': np.nan, 'median_dcr': np.nan, 'min_dcr': np.nan, 'pct_above_4': np.nan}
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(real_data)
    distances, _ = nn.kneighbors(synthetic_data)
    dcr_values = distances.flatten()
    return {
        'mean_dcr': float(np.mean(dcr_values)),
        'median_dcr': float(np.median(dcr_values)),
        'min_dcr': float(np.min(dcr_values)),
        'pct_above_4': float(np.mean(dcr_values > 4.0) * 100)
    }

def calculate_js_divergence(real_data, synthetic_data):
    """
    Compute Jensen-Shannon divergence across features.

    Args:
        real_data: Real samples, shape (n_real, n_features).
        synthetic_data: Synthetic samples, shape (n_synth, n_features).

    Returns:
        float: Mean per-feature Jensen-Shannon divergence.
    """
    js_scores = []
    if real_data.size == 0 or synthetic_data.size == 0: return np.nan
    if np.any(np.isnan(real_data)) or np.any(np.isnan(synthetic_data)): return np.nan
    
    num_features = real_data.shape[1]
    for i in range(num_features):
        min_val = min(real_data[:, i].min(), synthetic_data[:, i].min())
        max_val = max(real_data[:, i].max(), synthetic_data[:, i].max())
        
        if min_val == max_val:
            js_scores.append(0.0)
            continue
        
        bins = np.linspace(min_val, max_val, 30)
        hist_real, _ = np.histogram(real_data[:, i], bins=bins)
        hist_synth, _ = np.histogram(synthetic_data[:, i], bins=bins)
        
        epsilon = 1e-10
        prob_real = hist_real + epsilon
        prob_synth = hist_synth + epsilon
        prob_real = prob_real / prob_real.sum()
        prob_synth = prob_synth / prob_synth.sum()
        
        js = jensenshannon(prob_real, prob_synth)
        js_scores.append(js)
    
    return float(np.mean(js_scores))

def compute_wasserstein_distance(real_data, synthetic_data):
    """
    Compute Wasserstein distance across features.

    Args:
        real_data: Real samples, shape (n_real, n_features).
        synthetic_data: Synthetic samples, shape (n_synth, n_features).

    Returns:
        float: Mean per-feature Wasserstein distance.
    """
    wd_scores = []
    if real_data.size == 0 or synthetic_data.size == 0:
        return np.nan
    for i in range(real_data.shape[1]):
        wd = wasserstein_distance(real_data[:, i], synthetic_data[:, i])
        wd_scores.append(wd)
    return float(np.mean(wd_scores))

def compute_mmd_multiclass(X_real, X_synthetic, y_real, y_synthetic):
    """
    Compute MMD separately for each class, then average.
    
    This ensures distributional fidelity is measured PER CLASS,
    not just globally.
    """
    from sklearn.metrics.pairwise import rbf_kernel
    
    classes = np.unique(y_real)
    mmd_per_class = {}
    
    for cls in classes:
        X_real_cls = X_real[y_real == cls]
        X_syn_cls = X_synthetic[y_synthetic == cls]
        
        if len(X_real_cls) < 10 or len(X_syn_cls) < 10:
            mmd_per_class[cls] = np.nan
            continue
        
        # RBF kernel MMD
        K_rr = rbf_kernel(X_real_cls, X_real_cls).mean()
        K_ss = rbf_kernel(X_syn_cls, X_syn_cls).mean()
        K_rs = rbf_kernel(X_real_cls, X_syn_cls).mean()
        
        mmd = K_rr + K_ss - 2 * K_rs
        mmd_per_class[cls] = mmd
    
    # Average across classes
    mmd_avg = np.nanmean(list(mmd_per_class.values()))
    
    return mmd_avg, mmd_per_class
  
