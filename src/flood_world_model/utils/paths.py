from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
STATIC_DIR = DATA_DIR / "static"

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

DYNAMIC_CORE_V1 = FEATURES_DIR / "dynamic_core.zarr"
DYNAMIC_CORE = FEATURES_DIR / "dynamic_core_v2.zarr"
STATIC_CORE = FEATURES_DIR / "static_v3.zarr"
STATIC_MASKS = FEATURES_DIR / "static_masks_v3.zarr"
TRAINING_DIR = FEATURES_DIR / "training_v3"

V0_CHECKPOINT = CHECKPOINT_DIR / "world_model_v0_best.pt"
V1_CHECKPOINT = CHECKPOINT_DIR / "world_model_v1_best.pt"
NORMALIZATION_PATH = TRAINING_DIR / "normalization.json"
TEST_INDEX_PATH = TRAINING_DIR / "test_indices.npy"
OUTPUT_PATH = OUTPUT_DIR