"""
Global configuration for the benchmark.
"""

from pathlib import Path

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
CSV_DIR = RESULTS_DIR / "csv"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

LOG_DIR = PROJECT_ROOT / "logs"

# ==========================================================
# RANDOMNESS
# ==========================================================

RANDOM_SEEDS = [42, 123, 2024, 777, 999]

# ==========================================================
# TRAINING SAMPLE SIZES
# ==========================================================
TRAINING_SIZES = [
    10,
    20,
    35,
    50,
    75,
    100,
    150,
    250,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    4000,
]
# ==========================================================
# DATASETS
# ==========================================================

DATASETS = {
    "adult": 1590,
    # We'll add more datasets later
}

# ==========================================================
# MODELS
# ==========================================================

TREE_MODELS = [
    "XGBoost",
    "LightGBM",
    "CatBoost"
]

FOUNDATION_MODELS = [
    "TabPFN"
]