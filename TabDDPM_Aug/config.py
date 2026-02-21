"""
Working datasets configuration - tested and verified
"""

DATASET_CONFIGS = {
    # Easy (IR < 5)
    'wine': {'tabddpm_epochs': 400, 'batch_size': 64, 'lr': 2e-4, 'n_seeds': 10, 'n_classes': 3, 'imbalance_ratio': 1.48},
    'vehicle': {'tabddpm_epochs': 300, 'batch_size': 128, 'lr': 2e-4, 'n_seeds': 10, 'n_classes': 4, 'imbalance_ratio': 1.1},
    'new-thyroid': {'tabddpm_epochs': 400, 'batch_size': 64, 'lr': 2e-4, 'n_seeds': 10, 'n_classes': 3, 'imbalance_ratio': 5.0},
    
    # Medium (IR 5-15)
    'glass': {'tabddpm_epochs': 400, 'batch_size': 64, 'lr': 2e-4, 'n_seeds': 10, 'n_classes': 6, 'imbalance_ratio': 8.44},
    'dermatology': {'tabddpm_epochs': 300, 'batch_size': 128, 'lr': 2e-4, 'n_seeds': 10, 'n_classes': 6, 'imbalance_ratio': 5.55},
    'cleveland': {'tabddpm_epochs': 400, 'batch_size': 128, 'lr': 2e-4, 'n_seeds': 10, 'n_classes': 5, 'imbalance_ratio': 12.31},
    
    # Hard (IR > 15)
    'car': {'tabddpm_epochs': 300, 'batch_size': 256, 'lr': 2e-4, 'n_seeds': 10, 'n_classes': 4, 'imbalance_ratio': 18.62},
    'automobile': {'tabddpm_epochs': 400, 'batch_size': 64, 'lr': 2e-4, 'n_seeds': 10, 'n_classes': 6, 'imbalance_ratio': 16.0},
    'ecoli': {'tabddpm_epochs': 400, 'batch_size': 128, 'lr': 2e-4, 'n_seeds': 10, 'n_classes': 8, 'imbalance_ratio': 8.6},
}

def get_config(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        raise KeyError(f"Dataset '{dataset_name}' not found. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]

DATASET_FILES = {
    'wine': {'filename': 'wine.csv', 'target_col': 'class', 'drop_cols': [], 'source': 'uci', 'uci_id': 109},
    'vehicle': {'filename': 'vehicle.csv', 'target_col': 'Class', 'drop_cols': [], 'source': 'uci', 'uci_id': 149},
    'new-thyroid': {'filename': 'new-thyroid.csv', 'target_col': 'Class', 'drop_cols': [], 'source': 'uci', 'uci_id': 102},
    'glass': {'filename': 'glass.csv', 'target_col': 'Type', 'drop_cols': [], 'source': 'uci', 'uci_id': 42},
    'dermatology': {'filename': 'dermatology.csv', 'target_col': 'class', 'drop_cols': [], 'source': 'uci', 'uci_id': 33},
    'cleveland': {'filename': 'cleveland.csv', 'target_col': 'num', 'drop_cols': [], 'source': 'uci', 'uci_id': 45},
    'car': {'filename': 'car.csv', 'target_col': 'class', 'drop_cols': [], 'source': 'uci', 'uci_id': 19},
    'automobile': {'filename': 'automobile.csv', 'target_col': 'symboling', 'drop_cols': [], 'source': 'uci', 'uci_id': 10},
    'ecoli': {'filename': 'ecoli.csv', 'target_col': 'class', 'drop_cols': [], 'source': 'uci', 'uci_id': 39},
}
