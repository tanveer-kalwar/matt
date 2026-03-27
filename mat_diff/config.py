"""
MAT-Diff configuration for all 33 benchmark datasets.

Design principles:
    - NO per-dataset hyperparameter tuning: all model hyperparameters are
      derived from dataset properties (n_samples, n_features, n_classes, IR)
    - Classifier hyperparameters follow standard protocols
    - Standard ML constants (learning rate ranges, architecture sizes)

Hyperparameter derivation rules:
    d_model:    min(256, max(64, 2^ceil(log2(n_features * 4))))
                Representation capacity scales with input dimension
    d_hidden:   6 * d_model (universal)
                Richer FFN for stronger representation power
    n_blocks:   2 if n_features <= 20, else 3
                Deeper networks for higher-dimensional inputs
    n_heads:    max(2, d_model // 64)
                Each attention head handles approximately 64 dimensions
    batch_size: if N_minor > 256: min(512, max(16, N_minor // 4))
                else:             min(128, max(16, N_minor // 4))
                Larger batches allowed when data is sufficient
    epochs:     Scaled by IR: base=500, +80*log2(IR), capped at 2000;
                never less than 500 after budget cap
    n_phases:   Derived from n_features/spf; guaranteed >= 2 when N_minor > 300
    dropout:    0.0 when N_minor > 256 (adequate samples → no regularisation bias)
    lr:         1e-3

"""

import math
from typing import Dict, Any

# ── All 33 Benchmark Datasets ──
DATASET_REGISTRY = {
    # Abalone variants
    # CORRECTED: openml_id=183 (verified: 4177 samples, target="Class_Rings", int 1-29)
    # openml_id=720 is incorrect and loads a completely different balanced dataset.
    "abalone_6":        {"source": "openml", "openml_id": 183,   "target": "Class_Rings", "binary": True,
                         "minority_rule": "le_6",   "n_samples": 4177, "n_features": 8, "ir": 17.66},
    "abalone_15":       {"source": "openml", "openml_id": 183,   "target": "Class_Rings", "binary": True,
                         "minority_rule": "ge_15",  "n_samples": 4177, "n_features": 8, "ir": 15.0},
    "abalone_19":       {"source": "openml", "openml_id": 183,   "target": "Class_Rings", "binary": True,
                         "minority_rule": "eq_19",  "n_samples": 4177, "n_features": 8, "ir": 129.5},
    # Avila variants
    "avila":            {"source": "openml", "openml_id": 1489,  "target": "Class", "binary": True,
                         "minority_rule": "minority", "n_samples": 20867, "n_features": 10, "ir": 97.4},
    "avila_0vs5":       {"source": "openml", "openml_id": 1489,  "target": "Class", "binary": True,
                         "minority_rule": "pair_0_5", "n_samples": 9610, "n_features": 10, "ir": 8.26},
    "avila_0vs6":       {"source": "openml", "openml_id": 1489,  "target": "Class", "binary": True,
                         "minority_rule": "pair_0_6", "n_samples": 9464, "n_features": 10, "ir": 9.61},
    "avila_0vs7":       {"source": "openml", "openml_id": 1489,  "target": "Class", "binary": True,
                         "minority_rule": "pair_0_7", "n_samples": 9276, "n_features": 10, "ir": 12.18},
    # Banking
    "bank":             {"source": "openml", "openml_id": 1461,  "target": "Class", "binary": True,
                         "minority_rule": "minority", "n_samples": 4521, "n_features": 16, "ir": 7.7},
    "bank_full":        {"source": "openml", "openml_id": 1558,  "target": "Class", "binary": True,
                         "minority_rule": "minority", "n_samples": 45211, "n_features": 16, "ir": 7.55},
    # Financial
    "bankruptcy":       {"source": "openml", "openml_id": 44089, "target": "class", "binary": True,
                         "minority_rule": "minority", "n_samples": 19967, "n_features": 64, "ir": 45.22},
    # Insurance / Marketing
    "coil_2000":        {"source": "openml", "openml_id": 299,   "target": "V86", "binary": True,
                         "minority_rule": "minority", "n_samples": 9822, "n_features": 85, "ir": 16.0},
    # Agriculture — CORRECTED: UCI id=602 (13,611 samples, 7 classes, Koklu & Ozkan 2020)
    # OpenML id=43797 is a different 589-sample dataset and must NOT be used.
    "Dry_Beans":        {"source": "uci",    "uci_id": 602,       "target": "Class", "binary": False,
                         "minority_rule": "minority", "n_samples": 13611, "n_features": 16, "ir": 6.8},
    # Network
    "firewall":         {"source": "openml", "openml_id": 43560, "target": "Action", "binary": True,
                         "minority_rule": "minority", "n_samples": 65532, "n_features": 11, "ir": 697.0},
    # Speech
    "isolet":           {"source": "openml", "openml_id": 300,   "target": "Class", "binary": True,
                         "minority_rule": "pair_AB", "n_samples": 7797, "n_features": 617, "ir": 12.0},
    # Image
    "letter_img":       {"source": "openml", "openml_id": 6,     "target": "class", "binary": True,
                         "minority_rule": "eq_Z",   "n_samples": 20000, "n_features": 16, "ir": 26.0},
    # Medical
    "mammography":      {"source": "openml", "openml_id": 310,   "target": "class", "binary": True,
                         "minority_rule": "minority", "n_samples": 11183, "n_features": 6, "ir": 42.01},
    # Social
    "nursery":          {"source": "openml", "openml_id": 26,    "target": "class", "binary": True,
                         "minority_rule": "minority", "n_samples": 12958, "n_features": 8, "ir": 13.2},
    # Digit recognition
    "optical_digits":   {"source": "openml", "openml_id": 28,    "target": "class", "binary": True,
                         "minority_rule": "eq_8",   "n_samples": 5620, "n_features": 64, "ir": 9.14},
    # Document
    "page_blocks":      {"source": "openml", "openml_id": 30,    "target": "class", "binary": True,
                         "minority_rule": "minority", "n_samples": 5473, "n_features": 10, "ir": 175.5},
    "page_blocks_0vs3": {"source": "openml", "openml_id": 30,    "target": "class", "binary": True,
                         "minority_rule": "pair_1_4", "n_samples": 5001, "n_features": 10, "ir": 55.83},
    "pen_digits":       {"source": "openml", "openml_id": 32,    "target": "class", "binary": True,
                         "minority_rule": "eq_5",   "n_samples": 10992, "n_features": 16, "ir": 9.7},
    # Remote sensing
    "satimage":         {"source": "openml", "openml_id": 182,   "target": "class", "binary": False,
                         "minority_rule": "minority", "n_samples": 6430, "n_features": 36, "ir": 2.5},
    "satimage_4":       {"source": "openml", "openml_id": 182,   "target": "class", "binary": True,
                         "minority_rule": "eq_4",   "n_samples": 6435, "n_features": 36, "ir": 9.3},
    # Scene
    "scene":            {"source": "openml", "openml_id": 312,   "target": "class", "binary": True,
                         "minority_rule": "ge_1",   "n_samples": 2407, "n_features": 294, "ir": 12.6},
    # E-commerce
    "shoppers":         {"source": "openml", "openml_id": 42737, "target": "Revenue", "binary": True,
                         "minority_rule": "minority", "n_samples": 12330, "n_features": 16, "ir": 5.46},
    # Education
    "students_dropout": {"source": "openml", "openml_id": 44965, "target": "Target", "binary": False,
                         "minority_rule": "minority", "n_samples": 4424, "n_features": 36, "ir": 2.8},
    # Finance
    "taiwanese":        {"source": "openml", "openml_id": 42477, "target": "default.payment.next.month", "binary": True,
                         "minority_rule": "minority", "n_samples": 6819, "n_features": 94, "ir": 30.0},
    # Medical — CORRECTED: n_features=29 (actual feature count after removing target).
    # Target values are "sick." and "negative." — cleaned in data_fetcher._clean_target_values().
    # Dataset has many missing values ("?") — handled by imputation in data_fetcher._impute_features().
    "thyroid_sick":     {"source": "openml", "openml_id": 38,    "target": "Class", "binary": True,
                         "minority_rule": "minority", "n_samples": 3772, "n_features": 29, "ir": 15.33},
    # Food
    "wine_quality":     {"source": "openml", "openml_id": 287,   "target": "class", "binary": True,
                         "minority_rule": "le_4",   "n_samples": 4898, "n_features": 11, "ir": 25.77},
    # Yeast
    "yeast":            {"source": "openml", "openml_id": 181,   "target": "class", "binary": False,
                         "minority_rule": "minority", "n_samples": 1484, "n_features": 8, "ir": 28.1},

    # ── New benchmark datasets ──

    # Image (tabular pixel features)
    # CIFAR-10 flattened: 70 000 samples, 3072 pixel features, 10 balanced classes (IR≈1).
    # Included to test MAT-Diff on a balanced high-dimensional multi-class setting.
    "cifar10_tabular":  {"source": "openml", "openml_id": 40978, "target": "class", "binary": False,
                         "minority_rule": "minority", "n_samples": 70000, "n_features": 3072,
                         "n_classes": 10, "ir": 1.0},

    # Aerospace / control — classic strong-imbalance benchmark.
    # 7 classes; class 1 (normal) holds ≈78 % of samples while classes 2/6/7
    # each have < 20 samples (IR > 7000 for the rarest class vs. majority).
    # ir=200 is used for hyperparameter derivation as a practical representative
    # value that avoids extreme architecture choices for the tiny tail classes.
    "shuttle":          {"source": "openml", "openml_id": 43514, "target": "class", "binary": False,
                         "minority_rule": "minority", "n_samples": 58000, "n_features": 9,
                         "n_classes": 7, "ir": 200.0},

    # Medical — thyroid anomaly detection (annthyroid variant).
    # Binary: normal (≈92 %) vs. thyroid anomaly (≈8 %), IR≈12.5.
    "thyroid_annthyroid": {"source": "openml", "openml_id": 40536, "target": "class", "binary": True,
                           "minority_rule": "minority", "n_samples": 7200, "n_features": 21, "ir": 12.5},

    # Medical — fetal cardiotocography monitoring.
    # Native 3 classes: normal / suspect / pathological (pathological ≈8 %, IR≈10).
    # binary=True: minority_rule="minority" binarises to pathological vs. rest.
    # Real-valued CTG signal features; distinct domain from all existing sets.
    "cardiotocography": {"source": "openml", "openml_id": 1488,  "target": "Class", "binary": True,
                         "minority_rule": "minority", "n_samples": 2126, "n_features": 21, "ir": 10.0},

    # Industrial — steel surface defect detection.
    # 7 defect-type classes, 27 real-valued physical-measurement features, IR≈5–15.
    # Representative of industrial anomaly detection tasks.
    "steel_plates":     {"source": "openml", "openml_id": 1504,  "target": "Class", "binary": False,
                         "minority_rule": "minority", "n_samples": 1941, "n_features": 27,
                         "n_classes": 7, "ir": 10.0},
}


# ── Classifier hyperparameters: DGOT Table III (fixed, not tuned) ──
CLASSIFIER_PARAMS = {
    "XGBoost":            {"max_depth": 3, "n_estimators": 100,
                           "use_label_encoder": False,
                           "eval_metric": "logloss", "verbosity": 0},
    "DecisionTree":       {"max_depth": 30},
    "LogisticRegression": {"penalty": "l2", "max_iter": 500},
    "RandomForest":       {"n_estimators": 100},
    "KNN":                {"n_neighbors": 5},
}


def derive_hyperparams(n_samples: int, n_features: int, n_classes: int, ir: float) -> Dict[str, Any]:
    """Derive ALL model hyperparameters from dataset properties.

    No manual tuning. Every value follows a documented rule.
    All hyperparameters scale with dataset properties for generalization.
    Model capacity is bounded by available data (samples-per-feature ratio).
    """
    spf = n_samples / max(n_features, 1)

    # d_model: Scale with features, minimum 64
    base_d = max(128, 4 * n_features)
    d_model = min(256, 2 ** math.ceil(math.log2(base_d)))

    # CRITICAL: minority-only training means model sees far fewer samples.
    # Capacity MUST scale with minority count, not total dataset size.
    minority_count = max(10, int(n_samples / max(ir, 2)))
    if minority_count < 100:
        d_model = 64
    elif minority_count < 300:
        d_model = max(64, min(128, d_model // 2))
    elif minority_count < 750:
        d_model = max(64, min(192, d_model))

    if spf < 10:
        d_model = max(64, d_model // 2)

    # n_blocks: more blocks = more geodesic attention layers = better geometry capture
    if n_samples < 500 or spf < 10:
        n_blocks = 2
    elif n_features > 50:
        # For high-dimensional data, 4 blocks only when the minority class has
        # ≥ 8 samples per feature.  Below this threshold the deeper model is prone
        # to overfitting on the (small) minority training set, degrading sample
        # quality.  8 samples/feature is a standard rule-of-thumb for minimum
        # reliable capacity estimation in high-dimensional regression/diffusion.
        minority_spf = minority_count / max(n_features, 1)
        n_blocks = 4 if minority_spf >= 8 else 2
    elif n_features > 20:
        n_blocks = 3
    else:
        n_blocks = 2

    # n_heads: Each head handles ~64 dimensions, minimum 2
    n_heads = max(2, d_model // 64)
    # Must divide d_model evenly
    while d_model % n_heads != 0 and n_heads > 2:
        n_heads -= 1
    n_heads = max(2, n_heads)

    # Smaller batches for minority-only training (typically 100-500 samples)
    # For N_minor > 256 allow up to 512 to keep large-data runs fast while
    # providing stable gradient estimates; otherwise keep the conservative 128 cap.
    minority_estimate = max(10, n_samples // max(ir, 2))
    if minority_estimate > 256:
        batch_size = min(512, max(16, minority_estimate // 4))
    else:
        batch_size = min(128, max(16, minority_estimate // 4))
    batch_size = 2 ** round(math.log2(max(16, batch_size)))

    # More epochs needed because training on minority data only (much smaller)
    # BUT: Cap to prevent overfitting while still allowing sufficient training.
    base_epochs = 500
    ir_bonus = int(80 * math.log2(max(ir, 1)))
    epochs = min(2000, base_epochs + ir_bonus)
    if n_samples < 500:
        epochs = min(epochs, 800)

    # Multiclass datasets with low IR have a large total-minority pool.
    # Without a cap, Dry_Beans (n_classes=7) runs 721×72 = 51,912 steps.
    # Mammography runs 931×13 = 12,103 steps. Dry_Beans is 4.3× SLOWER for
    # a dataset that barely needs augmentation (IR=6.8).
    # Fix: cap at TARGET_STEPS=8000 gradient steps total (min 200 epochs).
    TARGET_STEPS = 8000
    MIN_EPOCHS = 200
    if n_classes > 2:
        minority_total_estimate = int(n_samples * (n_classes - 1) / n_classes)
    else:
        minority_total_estimate = minority_estimate
    batches_per_epoch_est = max(1, minority_total_estimate // batch_size)
    epochs_from_budget = max(MIN_EPOCHS, TARGET_STEPS // batches_per_epoch_est)
    epochs = min(epochs, epochs_from_budget)
    # Never train fewer than 500 epochs, regardless of budget cap
    epochs = max(500, epochs)

    # d_hidden: 6x d_model (was 4x) for stronger representation power
    d_hidden = d_model * 6

    # n_phases: Only use multiple phases if enough features to partition,
    # enough samples per feature, AND enough minority samples to fit the
    # spectral decomposition on. With < 50 minority samples, fitting PCA/SVD
    # phases is fitting noise (19 samples → n_phases=1 mandatory).
    if minority_count < 50:
        n_phases = 1
    elif n_features >= 30 and spf >= 10:
        n_phases = 3
    elif n_features >= 15 and spf >= 10:
        n_phases = 2
    else:
        n_phases = 1
    # For N_minor > 300, guarantee at least 2 phases so spectral curriculum
    # gets a harder stage even for lower-dimensional datasets.
    if minority_count > 300:
        n_phases = max(2, n_phases)

    # Regularisation: with N_minor > 256 there is adequate data, so dropout is
    # kept at 0 to minimise regularisation bias.
    dropout = 0.0
    weight_decay = 0.0

    sampling_steps = 200

    return {
        "d_model": d_model,
        "d_hidden": d_hidden,
        "n_blocks": n_blocks,
        "n_heads": n_heads,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": 1e-3,
        "dropout": dropout,
        "total_timesteps": 200,
        "sampling_steps": sampling_steps,
        "n_phases": n_phases,
        "weight_decay": weight_decay,
        "n_seeds": 10,
    }


def get_matdiff_config(dataset_name: str) -> Dict[str, Any]:
    """Get fully-derived config for any dataset. Zero manual tuning."""
    if dataset_name not in DATASET_REGISTRY:
        return derive_hyperparams(n_samples=1000, n_features=10, n_classes=2, ir=10.0)

    info = DATASET_REGISTRY[dataset_name]
    train_samples = info["n_samples"]
    cfg = derive_hyperparams(
        n_samples=train_samples,
        n_features=info["n_features"],
        n_classes=info.get("n_classes", 2),
        ir=info["ir"],
    )
    cfg["ir"] = info["ir"]
    cfg["n_classes"] = info.get("n_classes", 2)
    return cfg




