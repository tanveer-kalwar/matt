"""
SMOTE and ADASYN baseline generators.
"""
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

def smote_adasyn_generator(X_train_norm, y_train, generator_class, seed=42):
    """
    Apply SMOTE or ADASYN to balance the dataset.

    Args:
        X_train_norm: Training features.
        y_train: Training labels.
        generator_class: Oversampler class (SMOTE or ADASYN).
        seed: Random seed for the sampler.

    Returns:
        tuple: (X_resampled, y_resampled) after oversampling.
    """
    if not SMOTE_AVAILABLE:
        return X_train_norm, y_train
    print(f"    Applying {generator_class.__name__}...")
    try:
        sampler = generator_class(random_state=seed)
        X_resampled, y_resampled = sampler.fit_resample(X_train_norm, y_train)
        print(f"    Resampled: {len(X_train_norm)} → {len(X_resampled)}")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"    {generator_class.__name__} failed: {e}. Returning original data.")
        return X_train_norm, y_train
