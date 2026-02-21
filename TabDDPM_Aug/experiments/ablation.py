"""
Ablation study variant generators for strategy comparison.
"""
import numpy as np
from ..generators.tabddpm_aug import find_hard_samples, tabddpm_aug_ensemble_generator, dcr_filtering
from imblearn.over_sampling import SMOTE
SMOTE_AVAILABLE = True


def generate_ablation_variant(ablation_config, data, main_config, seed, device):
    """
    Generate synthetic data using a specific ablation configuration.
    
    Args:
        ablation_config: Dict with 'split_strategy' and 'filter_mode'
        data: Prepared dataset from prepare_data()
        main_config: Model hyperparameters
        seed: Random seed
        device: 'cuda' or 'cpu'
    """
    print(f"\n[{ablation_config['name']}]")
    
    # Unpack data
    X_train_norm = data['X_train_norm']
    y_train = data['y_train']
    X_minority = data['X_minority']
    minority_class = data['minority_class']
    n_needed = data['n_needed']
    n_features = X_train_norm.shape[1]
    n_minority = len(X_minority)

    # ========================================================================
    # 1. DBHA Split Strategy
    # ========================================================================
    X_minority_hard, X_minority_easy = X_minority, X_minority

    split_strategy = ablation_config['split_strategy']

    if split_strategy == 'adaptive':
        print("Using Adaptive Split Strategy...")

        # Use sample-to-feature ratio ρ = N_min / d for regime selection
        d = data['X_train_norm'].shape[1]
        rho = n_minority / max(d, 1)
        print(f"  Features d = {d}, density ratio ρ = {rho:.2f}")

        tau_size = 30.0  # density threshold used to separate low vs high-density 

        if rho < tau_size:
            print("  Low-density regime (DBHA): using Easy–Hard split")
            X_minority_hard, X_minority_easy = find_hard_samples(
                data['X_train_norm'], data['y_train'], data['X_minority'], data['minority_class'], seed
            )
        else:
            print("  High-density regime (FDHA): using full minority data")
            X_minority_hard = data['X_minority']
            X_minority_easy = data['X_minority']

    elif split_strategy == 'forced_dbha':
        print("Using Fixed DBHA (Easy–Hard split)...")
        X_minority_hard, X_minority_easy = find_hard_samples(
            data['X_train_norm'], data['y_train'], data['X_minority'], data['minority_class'], seed
        )
    else:
        print("Using Fixed FDHA (full data)...")
        X_minority_hard = data['X_minority']
        X_minority_easy = data['X_minority']
        
    # ========================================================================
    # 2. Define Ratio
    # ========================================================================
    # We'll use the "SMALL" dataset strategy for all ablations for consistency
    smote_ratio = 0.50
    n_smote = int(n_needed * smote_ratio)
    n_tabddpm = n_needed - n_smote

    # ========================================================================
    # 3. SMOTE Generation (on Easy samples)
    # ========================================================================
    print(f"\n    Step 2: Generating {n_smote} SMOTE samples from {len(X_minority_easy)} easy samples...")
    X_smote = np.array([]).reshape(0, n_features)
    if SMOTE_AVAILABLE and n_smote > 0 and len(X_minority_easy) > 1:
        try:
            majority_class = -1
            if minority_class == 0: majority_class = 1
            else: majority_class = 0
            
            X_majority = X_train_norm[y_train == majority_class]
            X_temp = np.vstack([X_minority_easy, X_majority[:min(len(X_majority), len(X_minority_easy)*2)]])
            y_temp = np.hstack([
                np.full(len(X_minority_easy), minority_class),
                np.full(min(len(X_majority), len(X_minority_easy)*2), majority_class)
            ])
            smote = SMOTE(
                sampling_strategy={minority_class: len(X_minority_easy) + n_smote},
                k_neighbors=min(5, len(X_minority_easy) - 1),
                random_state=seed
            )
            X_resampled, y_resampled = smote.fit_resample(X_temp, y_temp)
            X_smote = X_resampled[len(X_temp):len(X_temp) + n_smote]
        except Exception as e:
            print(f"    SMOTE failed: {e}.")
            indices = np.random.choice(len(X_minority_easy), n_smote, replace=True)
            X_smote = X_minority_easy[indices]
            
    # ========================================================================
    # 4. Diffusion Generation (on Hard samples)
    # ========================================================================
    X_tabddpm_pool = np.array([]).reshape(0, n_features)
    if n_tabddpm > 0 and len(X_minority_hard) > 1:
        print(f"\n    Step 3: Generating diffusion pool from {len(X_minority_hard)} hard samples...")
        
        # Generate 3x pool for filtering
        X_tabddpm_pool = tabddpm_aug_ensemble_generator(
            X_minority_hard, n_tabddpm, main_config, seed, device, n_models=3
        )
        
    # ========================================================================
    # 5. Filtering Step ---
    # ========================================================================            
    X_tabddpm_final = np.array([]).reshape(0, n_features)
    if len(X_tabddpm_pool) > 0:
        print(f"\n    Step 4: Filtering {len(X_tabddpm_pool)} pool samples...")
        filter_mode = ablation_config['filter_mode']
        
        if filter_mode == 'dcr':
            X_tabddpm_final = dcr_filtering(
                X_tabddpm_pool, X_minority, n_tabddpm, seed
            )
        else: # 'none'
            print("    Using No Filter (Random Sampling)...")
            np.random.seed(seed)
            indices = np.random.permutation(len(X_tabddpm_pool))
            X_tabddpm_final = X_tabddpm_pool[indices[:n_tabddpm]]
        
        print(f"    Selected {len(X_tabddpm_final)} diffusion samples.")

    # ========================================================================
    # 6. Combine
    # ========================================================================
    if len(X_smote) > 0 and len(X_tabddpm_final) > 0:
        X_combined = np.vstack([X_smote, X_tabddpm_final])
    elif len(X_smote) > 0:
        X_combined = X_smote
    elif len(X_tabddpm_final) > 0:
        X_combined = X_tabddpm_final
    else:
        return None # Generation failed
    
    # ========================================================================
    # 7. Finalize Count
    # ========================================================================
    if len(X_combined) > n_needed:
        np.random.seed(seed)
        indices = np.random.permutation(len(X_combined))
        X_final = X_combined[indices[:n_needed]]
    elif len(X_combined) < n_needed and len(X_combined) > 0:
        additional = n_needed - len(X_combined)
        indices = np.random.choice(len(X_combined), size=additional, replace=True)
        X_final = np.vstack([X_combined, X_combined[indices]])
    else:
        X_final = X_combined
        
    return X_final
