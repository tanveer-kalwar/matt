"""
Dataset loading and preprocessing utilities with automatic download.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from .config import DATASET_FILES


def download_dataset(dataset_name):
    """
    Automatically download dataset from UCI or OpenML if not present.
    
    Args:
        dataset_name: Name of the dataset to download
        
    Returns:
        pandas.DataFrame: Downloaded dataset
    """
    config = DATASET_FILES[dataset_name]
    
    if config['source'] == 'manual':
        return None  # User must provide these manually
    
    print(f"  Downloading {dataset_name} from {config['source'].upper()}...")
    
    if config['source'] == 'uci':
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=config['uci_id'])
            df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            
            # Save to data directory
            os.makedirs('data', exist_ok=True)
            df.to_csv(f"data/{config['filename']}", index=False)
            print(f"  ✓ Downloaded and saved to data/{config['filename']}")
            return df
            
        except ImportError:
            print(f"  ✗ ucimlrepo not installed. Run: pip install ucimlrepo")
            raise RuntimeError("Cannot download UCI datasets without ucimlrepo")
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            raise
    
    elif config['source'] == 'openml':
        try:
            import openml
            dataset = openml.datasets.get_dataset(config['openml_id'])
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
            df = pd.concat([X, y], axis=1)
            
            # Save to data directory
            os.makedirs('data', exist_ok=True)
            df.to_csv(f"data/{config['filename']}", index=False)
            print(f"  ✓ Downloaded and saved to data/{config['filename']}")
            return df
            
        except ImportError:
            print(f"  ✗ openml not installed. Run: pip install openml")
            raise RuntimeError("Cannot download OpenML datasets without openml")
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            raise
    
    return None


def load_dataset(dataset_name):
    """Load and preprocess a dataset by name with multi-class support.

    Args:
        dataset_name (str): Key identifying the dataset in DATASET_FILES.

    Returns:
        tuple: (pandas.DataFrame, int) - Preprocessed dataframe with 'target' column and number of classes
    """
    config = DATASET_FILES[dataset_name]
    
    # Try to load from local first
    df = None
    for path in [f'data/{config["filename"]}', config["filename"]]:
        if os.path.exists(path):
            df = pd.read_csv(path, skipinitialspace=True)
            print(f"  Loaded from {path}")
            break
    
    # If not found locally, try to download
    if df is None:
        if config['source'] in ['uci', 'openml']:
            df = download_dataset(dataset_name)
        else:
            raise FileNotFoundError(
                f"{config['filename']} not found in {os.getcwd()} or data/. "
                f"Please place {config['filename']} in the data/ directory."
            )
    
    # Clean columns
    df.columns = df.columns.str.strip()
    df = df.replace(['?', ' ?', '  ?', 'NA'], np.nan).dropna()
    
    # Handle target column
    target_col = config['target_col'] if config['target_col'] in df.columns else df.columns[-1]
    
    # Multi-class encoding
    if 'pos_labels' in config:
        # Binary classification (backward compatibility)
        pos_labels = [str(label) for label in config['pos_labels']]
        df['target'] = df[target_col].astype(str).str.strip().apply(
            lambda x: 1 if x in pos_labels else 0
        )
    else:
        # Multi-class classification
        le = LabelEncoder()
        df['target'] = le.fit_transform(df[target_col].astype(str).str.strip())
        config['label_encoder'] = le  # Store for potential inverse transform
    
    # Special handling for credit card dataset
    if dataset_name == 'credit':
        if 'Time' in df.columns:
            df = df.drop(columns=['Time'])
        if len(df) > 100000:
            print(f"  Subsampling Credit dataset to 50k for efficiency")
            df_maj = df[df['target'] == 0].sample(50000, random_state=42)
            df_min = df[df['target'] == 1]
            df = pd.concat([df_maj, df_min]).sample(frac=1, random_state=42)
    
    # Drop unwanted columns
    df = df.drop(columns=[target_col] + config['drop_cols'], errors='ignore')
    
    # Get number of classes
    n_classes = df['target'].nunique()
    class_dist = df['target'].value_counts().sort_index().to_dict()
    
    print(f"\n{dataset_name.capitalize()}: {df.shape}, Classes: {n_classes}")
    print(f"  Distribution: {class_dist}")
    
    return df, n_classes


def prepare_data(df, seed=42):
    """Split and preprocess dataset for multi-class training.

    Args:
        df (pandas.DataFrame): Input dataframe containing features and 'target' column.
        seed (int, optional): Random state for splitting and preprocessing. Defaults to 42.

    Returns:
        dict: Dictionary with normalized train/test arrays and class information:
            - 'X_train_norm', 'y_train'
            - 'X_test_norm', 'y_test'
            - 'X_minority', 'X_minority_df' (for backward compatibility)
            - 'class_distribution': {class_id: count}
            - 'minority_classes': {class_id: count} for all non-majority classes
            - 'majority_class', 'minority_class' (smallest class)
            - 'n_needed' (total samples needed for balancing)
            - 'categorical_cols', 'numeric_cols'
            - 'scaler', 'label_encoders', 'X_train_df_full'
    """
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df.drop('target', axis=1), df['target'], 
        test_size=0.2, random_state=seed, stratify=df['target']
    )
    
    # Get class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    
    if len(counts) < 2:
        print("Warning: Only one class found in training data.")
        minority_class = 0
        majority_class = 0
    else:
        minority_class = unique[np.argmin(counts)]  # Smallest class
        majority_class = unique[np.argmax(counts)]  # Largest class
    
    # Identify all minority classes (everything except majority)
    majority_count = counts.max()
    minority_classes = {
        cls: count for cls, count in class_distribution.items()
        if cls != majority_class
    }
    
    # Calculate total samples needed for full balancing
    n_needed = sum(max(0, majority_count - count) for count in class_distribution.values())
    
    # Feature preprocessing
    numeric_cols = list(X_train_df.select_dtypes(include=np.number).columns)
    categorical_cols = list(X_train_df.select_dtypes(include='object').columns)
    
    # Use QuantileTransformer for robustness to outliers
    scaler = QuantileTransformer(output_distribution='uniform', random_state=seed)
    
    # Handle empty numeric cols
    if not numeric_cols:
        X_train_norm_num = np.empty((len(X_train_df), 0))
        X_test_norm_num = np.empty((len(X_test_df), 0))
    else:
        X_train_norm_num = scaler.fit_transform(X_train_df[numeric_cols])
        X_test_norm_num = scaler.transform(X_test_df[numeric_cols])
    
    # Encode categorical features
    X_train_processed, X_test_processed, label_encoders = [], [], {}
    for col in categorical_cols:
        le = LabelEncoder()
        all_values = pd.concat([X_train_df[col], X_test_df[col]]).astype(str).unique()
        le.fit(all_values)
        X_train_processed.append(le.transform(X_train_df[col].astype(str)).reshape(-1, 1))
        X_test_processed.append(le.transform(X_test_df[col].astype(str)).reshape(-1, 1))
        label_encoders[col] = le
    
    # Combine numeric and categorical
    if X_train_processed:
        X_train_norm = np.hstack([X_train_norm_num, np.hstack(X_train_processed)])
        X_test_norm = np.hstack([X_test_norm_num, np.hstack(X_test_processed)])
    else:
        X_train_norm, X_test_norm = X_train_norm_num, X_test_norm_num
    
    # Extract minority class data (for backward compatibility)
    X_minority = X_train_norm[y_train.values == minority_class]
    X_minority_df = X_train_df[y_train.values == minority_class].reset_index(drop=True)
    
    print(f"\nClass Distribution (Training):")
    for cls, count in sorted(class_distribution.items()):
        print(f"  Class {cls}: {count} samples")
    print(f"Majority Class: {majority_class} ({majority_count} samples)")
    print(f"Minority Classes: {len(minority_classes)}")
    print(f"Total samples needed for balancing: {n_needed}")
    
    return {
        'X_train_norm': X_train_norm, 
        'y_train': y_train.values,
        'X_test_norm': X_test_norm, 
        'y_test': y_test.values,
        'X_minority': X_minority, 
        'X_minority_df': X_minority_df,
        'minority_class': minority_class,
        'majority_class': majority_class,
        'class_distribution': class_distribution,
        'minority_classes': minority_classes,
        'n_needed': n_needed,
        'categorical_cols': categorical_cols, 
        'numeric_cols': numeric_cols,
        'scaler': scaler, 
        'label_encoders': label_encoders, 
        'X_train_df_full': X_train_df
    }
