"""
GAN-based baseline generators (Copula, CTGAN, TVAE, CTAB-GAN+).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sys

try:
    from ctgan import CTGAN, TVAE
    CTGAN_AVAILABLE = True
    print("CTGAN/TVAE available")
except ImportError:
    CTGAN_AVAILABLE = False
    print("CTGAN/TVAE not available")

REPO_ROOT = Path(__file__).resolve().parents[2]
CTABGAN_PLUS_ROOT = REPO_ROOT / "CTAB-GAN-Plus"

if str(CTABGAN_PLUS_ROOT) not in sys.path:
    sys.path.insert(0, str(CTABGAN_PLUS_ROOT))

from model.ctabgan import CTABGAN as CTABGAN_P


def _transform_synthetic_data(X_synthetic_df, scaler, label_encoders, numeric_cols, categorical_cols):
    """
    Transform a synthetic DataFrame back to the normalized numpy array space.

    Args:
        X_synthetic_df: Synthetic samples as a pandas DataFrame.
        scaler: Fitted scaler used for numerical feature normalization.
        label_encoders: Dict mapping column name to fitted LabelEncoder.
        numeric_cols: List of numerical feature column names.
        categorical_cols: List of categorical feature column names.

    Returns:
        ndarray: Normalized synthetic data, shape (n_samples, n_features).
    """
    # Ensure all numeric columns are present, fill with 0 if not
    for col in numeric_cols:
        if col not in X_synthetic_df.columns:
            X_synthetic_df[col] = 0.0

    X_synthetic_norm_num = np.empty((len(X_synthetic_df), 0))
    if numeric_cols:
        X_synthetic_norm_num = scaler.transform(X_synthetic_df[numeric_cols].values)
        
    X_synthetic_processed_cat = []
    
    for col in categorical_cols:
        le = label_encoders[col]
        
        if col not in X_synthetic_df.columns:
             X_synthetic_df[col] = le.classes_[0] 

        synthetic_labels = X_synthetic_df[col].astype(str).unique()
        seen_labels = [label for label in synthetic_labels if label in le.classes_]
        unseen_labels = [label for label in synthetic_labels if label not in le.classes_]
        
        if unseen_labels:
            # Find most frequent *seen* label to replace unseen ones
            if seen_labels:
                most_frequent_seen = pd.Series(seen_labels).mode()[0]
            else:
                # If no labels are seen (edge case), just use the first known class
                most_frequent_seen = le.classes_[0]
            
            X_synthetic_df[col] = X_synthetic_df[col].apply(lambda x: x if x in le.classes_ else most_frequent_seen)

        encoded_col = le.transform(X_synthetic_df[col].astype(str)).reshape(-1, 1)
        X_synthetic_processed_cat.append(encoded_col)

    if X_synthetic_processed_cat:
        X_synthetic_norm_cat = np.hstack(X_synthetic_processed_cat)
        X_synthetic_norm = np.hstack([X_synthetic_norm_num, X_synthetic_norm_cat])
    else:
        X_synthetic_norm = X_synthetic_norm_num
    return X_synthetic_norm

def copula_generator(X_minority_df, n_needed, categorical_cols, scaler, label_encoders, numeric_cols, seed=42):
    """Generate synthetic data using GaussianCopula.

    Args:
        X_minority_df (pandas.DataFrame): Minority-class samples.
        n_needed (int): Number of synthetic samples to generate.
        categorical_cols (list[str]): Names of categorical columns.
        scaler: Fitted scaler for numerical features.
        label_encoders (dict): Mapping column name -> fitted LabelEncoder.
        numeric_cols (list[str]): Names of numerical columns.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        numpy.ndarray: Normalized synthetic samples of shape (n_needed, n_features).
    """
    try:
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata
    except ImportError:
        raise RuntimeError("GaussianCopula not available")
    
    print(f"        Training GaussianCopula on {len(X_minority_df)} samples...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(X_minority_df)
    for col in categorical_cols:
        if col in X_minority_df.columns:
            metadata.update_column(column_name=col, sdtype='categorical')
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(X_minority_df)
    X_synthetic_df = synthesizer.sample(n_needed)
    X_synthetic_norm = _transform_synthetic_data(X_synthetic_df, scaler, label_encoders, numeric_cols, categorical_cols)
    print(f"    Generated {len(X_synthetic_norm)} samples")
    return X_synthetic_norm

def ctgan_generator(X_minority_df, X_minority_norm, n_needed, categorical_cols, scaler, label_encoders, numeric_cols, seed=42):
    """Generate synthetic data using CTGAN.

    Args:
        X_minority_df (pandas.DataFrame): Minority-class samples.
        X_minority_norm (numpy.ndarray): Normalized minority samples used for training.
        n_needed (int): Number of synthetic samples to generate.
        categorical_cols (list[str]): Names of categorical columns.
        scaler: Fitted scaler for numerical features.
        label_encoders (dict): Mapping column name -> fitted LabelEncoder.
        numeric_cols (list[str]): Names of numerical columns.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        numpy.ndarray: Normalized synthetic samples of shape (n_needed, n_features).
    """
    if not CTGAN_AVAILABLE:
        raise RuntimeError("CTGAN not available")
    print(f"        Training CTGAN on {len(X_minority_df)} samples...")
    synthesizer = CTGAN(embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256), 
                        epochs=50, cuda=torch.cuda.is_available())
    synthesizer.fit(X_minority_df, discrete_columns=categorical_cols)
    X_synthetic_df = synthesizer.sample(n_needed)
    X_synthetic_norm = _transform_synthetic_data(X_synthetic_df, scaler, label_encoders, numeric_cols, categorical_cols)
    print(f"    Generated {len(X_synthetic_norm)} samples")
    return X_synthetic_norm

def tvae_generator(X_minority_df, X_minority_norm, n_needed, categorical_cols, scaler, label_encoders, numeric_cols, seed=42):
    """Generate synthetic data using TVAE.

    Args:
        X_minority_df (pandas.DataFrame): Minority-class samples.
        X_minority_norm (numpy.ndarray): Normalized minority samples used for training TVAE.
        n_needed (int): Number of synthetic samples to generate.
        categorical_cols (list[str]): Names of categorical columns.
        scaler: Fitted scaler for numerical features.
        label_encoders (dict): Mapping column name -> fitted LabelEncoder.
        numeric_cols (list[str]): Names of numerical columns.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        numpy.ndarray: Normalized synthetic samples of shape (n_needed, n_features).
    """
    if not CTGAN_AVAILABLE:
        raise RuntimeError("TVAE not available")
    print(f"        Training TVAE on {len(X_minority_df)} samples...")
    synthesizer = TVAE(embedding_dim=128, compress_dims=(128, 128), decompress_dims=(128, 128), 
                       epochs=50, cuda=torch.cuda.is_available())
    synthesizer.fit(X_minority_df, discrete_columns=categorical_cols)
    X_synthetic_df = synthesizer.sample(n_needed)
    X_synthetic_norm = _transform_synthetic_data(X_synthetic_df, scaler, label_encoders, numeric_cols, categorical_cols)
    print(f"    Generated {len(X_synthetic_norm)} samples")
    return X_synthetic_norm

def ctabgan_plus_generator(X_minority_df, X_minority_norm, n_needed, categorical_cols, scaler, label_encoders, numeric_cols, minority_class, seed=42):
    """Generate synthetic data using CTAB-GAN+.

    Args:
        X_minority_df (pandas.DataFrame): Minority-class samples.
        X_minority_norm (numpy.ndarray): Normalized minority samples used for training CTAB-GAN+.
        n_needed (int): Number of synthetic samples to generate.
        categorical_cols (list[str]): Names of categorical columns.
        scaler: Fitted scaler for numerical features.
        label_encoders (dict): Mapping column name -> fitted LabelEncoder.
        numeric_cols (list[str]): Names of numerical columns.
        minority_class: Label value corresponding to the minority class.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        numpy.ndarray: Normalized synthetic samples of shape (n_needed, n_features).
    """
    if not 'model.ctabgan' in sys.modules:
        raise RuntimeError("CTAB-GAN-Plus not available")
    print(f"        Training CTAB-GAN-Plus on {len(X_minority_df)} samples...")
    df_fit = X_minority_df.copy()
    df_fit['target'] = minority_class
    cat_fit_cols = [col for col in categorical_cols if col in df_fit.columns]
    cat_fit_cols.append('target')
    problem_type = {'Classification': 'target'}
    synthesizer = CTABGAN_P(df=df_fit, test_ratio=0.0, categorical_columns=cat_fit_cols,
                          mixed_columns={}, general_columns=[], non_categorical_columns=[],
                          integer_columns=[], problem_type=problem_type, epochs=50,
                          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    synthesizer.fit()
    X_synthetic_df_full = synthesizer.generate_samples(n_needed)
    X_synthetic_df = X_synthetic_df_full.drop('target', axis=1)
    X_synthetic_norm = _transform_synthetic_data(X_synthetic_df, scaler, label_encoders, numeric_cols, categorical_cols)
    print(f"    Generated {len(X_synthetic_norm)} samples")
    return X_synthetic_norm
