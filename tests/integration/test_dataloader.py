from torch.utils.data import DataLoader

from flood_world_model.datasets.world_model_dataset_v2 import FloodWorldModelDataset

print("=" * 80)
print("TESTING WORLD MODEL DATALOADER")
print("=" * 80)

dataset = FloodWorldModelDataset(
    split="train",
    history_days=14,
    forecast_days=1,
)

print("Dataset samples:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

print("Loading one batch...")

batch = next(iter(loader))

print("Batch returned:", len(batch))

x, static, y, mask = batch

print("\nShapes:")
print("X      :", x.shape)
print("Static :", static.shape)
print("Y      :", y.shape)
print("Mask   :", mask.shape)

print("\nDtypes:")
print("X      :", x.dtype)
print("Static :", static.dtype)
print("Y      :", y.dtype)
print("Mask   :", mask.dtype)

print("\nFinite checks:")
print("X      :", bool(x.isfinite().all()))
print("Static :", bool(static.isfinite().all()))
print("Y      :", bool(y.isfinite().all()))
print("Mask   :", bool(mask.isfinite().all()))

print(
    "\nTarget valid fraction:",
    float(mask.mean()),
)

print("\nValue ranges:")
print(
    "X      :",
    float(x.min()),
    "->",
    float(x.max()),
)

print(
    "Static :",
    float(static.min()),
    "->",
    float(static.max()),
)

print(
    "Y      :",
    float(y.min()),
    "->",
    float(y.max()),
)

print(
    "\n✅ DATALOADER TEST PASSED"
)
