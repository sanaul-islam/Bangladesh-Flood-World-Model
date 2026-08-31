from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DYNAMIC_PATH = (
    PROJECT_ROOT
    / "data/features/dynamic_core_v2.zarr"
)

STATIC_PATH = (
    PROJECT_ROOT
    / "data/features/static_v3.zarr"
)

CHECKPOINT_V0 = (
    PROJECT_ROOT
    / "checkpoints/world_model_v0_best.pt"
)

CHECKPOINT_V2 = (
    PROJECT_ROOT
    / "checkpoints/world_model_v2_best.pt"
)

HISTORY_LENGTH = 14
HORIZON = 7

BATCH_SIZE = 2
HIDDEN_CHANNELS = 12

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15