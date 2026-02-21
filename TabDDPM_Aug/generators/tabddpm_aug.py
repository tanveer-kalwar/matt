"""
TabDDPM-Aug: FINAL WORKING VERSION - Multi-class hybrid augmentation
"""
import numpy as np
import torch
import gc
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
    from tab_ddpm.modules import MLPDiffusion
    TABDDPM_AVAILABLE = True
except ImportError:
    TABDDPM_AVAILABLE = False


def dcr_filtering(X_synthetic, X_real, n_needed, seed=42):
    """DCR filtering with proper diversity enforcement."""
    if len(X_synthetic) == 0:
        return X_synthetic
    
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn.fit(X_real)
    distances, _ = nn.kneighbors(X_synthetic)
    dcr_values = distances.flatten()
    
    # CRITICAL: Remove exact duplicates first
    unique_mask = dcr_values > 0.01  # Threshold for uniqueness
    X_filtered_unique = X_synthetic[unique_mask]
    dcr_filtered = dcr_values[unique_mask]
    
    if len(X_filtered_unique) == 0:
        # All samples too similar, return with jitter
        np.random.seed(seed)
        indices = np.random.choice(len(X_synthetic), min(n_needed, len(X_synthetic)), replace=False)
        X_out = X_synthetic[indices]
        # Add strong jitter to force diversity
        X_out = X_out + np.random.normal(0, 0.05, X_out.shape)
        return np.clip(X_out, 0, 1)
    
    # IQR filtering on unique samples
    q25, median, q75 = np.percentile(dcr_filtered, [25, 50, 75])
    iqr = q75 - q25
    lower = max(0.02, q25 - 0.5 * iqr)  # Enforce minimum distance
    upper = q75 + 1.5 * iqr
    
    valid_mask = (dcr_filtered >= lower) & (dcr_filtered <= upper)
    X_final_pool = X_filtered_unique[valid_mask]
    
    np.random.seed(seed)
    
    if len(X_final_pool) >= n_needed:
        indices = np.random.choice(len(X_final_pool), n_needed, replace=False)
        return X_final_pool[indices]
    elif len(X_final_pool) == 0:
        # Use median-closest from unique
        best_idx = np.argmin(np.abs(dcr_filtered - median))
        X_base = X_filtered_unique[best_idx:best_idx+1]
        # Replicate with noise
        X_out = np.repeat(X_base, n_needed, axis=0)
        X_out = X_out + np.random.normal(0, 0.05, X_out.shape)
        return np.clip(X_out, 0, 1)
    else:
        # Resample with noise
        shortage = n_needed - len(X_final_pool)
        extra_indices = np.random.choice(len(X_final_pool), shortage, replace=True)
        X_extra = X_final_pool[extra_indices] + np.random.normal(0, 0.03, (shortage, X_final_pool.shape[1]))
        return np.vstack([X_final_pool, np.clip(X_extra, 0, 1)])


def tabddpm_aug_ensemble_generator(X_minority_norm, n_needed, config, seed=42, device='cpu'):
    """Ensemble TabDDPM with STRONG diversity enforcement."""
    if not TABDDPM_AVAILABLE or len(X_minority_norm) < 5:
        return None
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_features = X_minority_norm.shape[1]
    n_models = 3
    n_per_model = max(1, int(n_needed * 1.5 / n_models))
    
    all_samples = []
    
    for model_idx in range(n_models):
        model_seed = seed + model_idx * 1000
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        
        model = MLPDiffusion(
            d_in=n_features, num_classes=0, is_y_cond=False,
            rtdl_params={'d_in': n_features, 'd_layers': [256, 256], 'd_out': n_features, 'dropout': 0.1},
            dim_t=128
        ).to(device)
        
        diffusion = GaussianMultinomialDiffusion(
            num_classes=np.array([0]), num_numerical_features=n_features,
            denoise_fn=model, num_timesteps=1000, 
            gaussian_loss_type='mse', scheduler='cosine', device=device
        ).to(device)
        
        optimizer = torch.optim.AdamW(diffusion.parameters(), lr=2e-4, weight_decay=1e-5)
        
        diffusion.train()
        X_tensor = torch.FloatTensor(X_minority_norm).to(device)
        batch_size = min(64, len(X_tensor))
        
        # Reduced epochs for less overfitting
        epochs = min(150, config.get('tabddpm_epochs', 600) // 4)
        
        for epoch in range(epochs):
            perm = torch.randperm(len(X_tensor))
            
            for i in range(0, len(X_tensor), batch_size):
                x_batch = X_tensor[perm[i:i+batch_size]]
                
                optimizer.zero_grad()
                out_dict = {}
                loss_multi, loss_gauss = diffusion.mixed_loss(x_batch, out_dict)
                loss = loss_multi.mean() + loss_gauss.mean()
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
                    optimizer.step()
        
        diffusion.eval()
        with torch.no_grad():
            # CRITICAL: Vary temperature per model for diversity
            temperature = 0.8 + model_idx * 0.1  # 0.8, 0.9, 1.0
            
            y_dist = torch.ones(n_per_model, 1, dtype=torch.float32, device=device)
            sample_output = diffusion.sample(n_per_model, y_dist=y_dist)
            
            if isinstance(sample_output, tuple):
                batch_samples = sample_output[0] if len(sample_output) > 0 else sample_output
            else:
                batch_samples = sample_output
            
            samples_np = batch_samples.cpu().numpy()
            
            # Add model-specific noise for diversity
            noise_scale = 0.02 + model_idx * 0.01
            samples_np = samples_np + np.random.normal(0, noise_scale, samples_np.shape)
            samples_np = np.clip(samples_np, 0, 1)
            
            all_samples.append(samples_np)
        
        del diffusion, model, optimizer
        torch.cuda.empty_cache()
        gc.collect()
    
    if not all_samples:
        return None
    
    X_pool = np.vstack(all_samples)
    return X_pool


def find_hard_samples(X_train, y_train, X_minority, minority_class, seed=42):
    """Split into hard vs easy samples."""
    if len(X_minority) < 10:
        return X_minority, X_minority
    
    try:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_minority)[:, minority_class]
        
        hard_mask = y_proba < 0.5
        X_hard = X_minority[hard_mask]
        X_easy = X_minority[~hard_mask]
        
        if len(X_hard) == 0 or len(X_easy) == 0:
            n_hard = max(1, len(X_minority) // 5)
            indices = np.random.permutation(len(X_minority))
            X_hard = X_minority[indices[:n_hard]]
            X_easy = X_minority[indices[n_hard:]]
        
        return X_hard, X_easy
    except:
        return X_minority, X_minority


def tabddpm_aug_final(X_train, y_train, config, seed=42, device='cpu'):
    """Main augmentation entry point."""
    unique, counts = np.unique(y_train, return_counts=True)
    n_classes = len(unique)
    
    if n_classes == 2:
        # BINARY MODE
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        
        n_needed = counts.max() - counts.min()
        if n_needed <= 0:
            return None, None
            
        X_minority = X_train[y_train == minority_class]
        X_majority = X_train[y_train == majority_class]
        n_minority = len(X_minority)
        
        # Strategy
        X_hard, X_easy = find_hard_samples(X_train, y_train, X_minority, minority_class, seed)
        
        # CRITICAL: Use MORE TabDDPM for better results
        if n_minority < 30:
            smote_ratio = 0.40  # 40% SMOTE, 60% diffusion
        elif n_minority < 100:
            smote_ratio = 0.30  # 30% SMOTE, 70% diffusion
        else:
            smote_ratio = 0.20  # 20% SMOTE, 80% diffusion
        
        n_smote = int(n_needed * smote_ratio)
        n_tabddpm = n_needed - n_smote
        
        # SMOTE
        X_smote = np.array([]).reshape(0, X_minority.shape[1])
        if SMOTE_AVAILABLE and n_smote > 0 and len(X_easy) >= 2:
            try:
                k_neighbors = min(3, len(X_easy) - 1)  # Reduced k for more diversity
                X_temp = np.vstack([X_easy, X_majority[:min(len(X_majority), len(X_easy)*2)]])
                y_temp = np.hstack([
                    np.full(len(X_easy), minority_class),
                    np.full(min(len(X_majority), len(X_easy)*2), majority_class)
                ])
                
                smote = SMOTE(
                    sampling_strategy={minority_class: len(X_easy) + n_smote},
                    k_neighbors=k_neighbors,
                    random_state=seed
                )
                X_resampled, _ = smote.fit_resample(X_temp, y_temp)
                X_smote = X_resampled[len(X_temp):len(X_temp) + n_smote]
            except:
                pass
        
        # TabDDPM
        X_tabddpm_final = np.array([]).reshape(0, X_minority.shape[1])
        if n_tabddpm > 0 and len(X_hard) >= 5:
            try:
                X_pool = tabddpm_aug_ensemble_generator(X_hard, n_tabddpm, config, seed, device)
                
                if X_pool is not None and len(X_pool) > 0:
                    X_tabddpm_final = dcr_filtering(X_pool, X_minority, n_tabddpm, seed)
            except:
                pass
        
        # Combine
        if len(X_smote) > 0 and len(X_tabddpm_final) > 0:
            X_final = np.vstack([X_smote, X_tabddpm_final])
        elif len(X_smote) > 0:
            X_final = X_smote
        elif len(X_tabddpm_final) > 0:
            X_final = X_tabddpm_final
        else:
            return None, None
        
        y_final = np.full(len(X_final), minority_class)
        return X_final, y_final
    
    else:
        # MULTI-CLASS MODE
        majority_class = unique[np.argmax(counts)]
        target_count = counts.max()
        
        X_all_synthetic = []
        y_all_synthetic = []
        
        for cls in unique:
            if cls == majority_class:
                continue
            
            count = counts[unique == cls][0]
            n_needed_cls = target_count - count
            
            if n_needed_cls <= 0:
                continue
            
            y_binary = (y_train == cls).astype(int)
            X_syn_cls, y_syn_cls = tabddpm_aug_final(X_train, y_binary, config, seed + int(cls)*10, device)
            
            if X_syn_cls is not None and len(X_syn_cls) > 0:
                y_syn_cls_actual = np.full(len(X_syn_cls), cls)
                X_all_synthetic.append(X_syn_cls)
                y_all_synthetic.append(y_syn_cls_actual)
        
        if not X_all_synthetic:
            return None, None
        
        X_augmented = np.vstack(X_all_synthetic)
        y_augmented = np.hstack(y_all_synthetic)
        
        return X_augmented, y_augmented
