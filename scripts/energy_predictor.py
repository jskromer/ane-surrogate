"""
Building Energy Predictor: PyTorch → CoreML → ANE

Surrogate model trained on EnergyPlus batch simulation data.
Predicts monthly electricity and natural gas consumption (kWh) from
building calibration parameters (wall insulation, infiltration, HVAC COP,
lighting density) and month of year.
"""
import csv
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path

# ── Feature and target definitions ────────────────────────────────────────

FEATURE_COLS = [
    "wall_insul_thickness",
    "infil_multiplier",
    "cooling_cop",
    "lighting_density",
    "month",
]

TARGET_COLS = [
    "electricity_kwh",
    "gas_kwh",
]

N_FEATURES = len(FEATURE_COLS)
N_TARGETS = len(TARGET_COLS)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "eplus_training_data.csv"

# ── Phase A: Load EnergyPlus simulation data ──────────────────────────────

def load_data(csv_path, test_fraction=0.2, seed=42):
    """Load CSV and split by run_id (avoids leaking correlated monthly data)."""
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    run_ids = sorted(set(int(r["run_id"]) for r in rows))
    rng = np.random.RandomState(seed)
    rng.shuffle(run_ids)

    n_test = max(1, int(len(run_ids) * test_fraction))
    test_ids = set(run_ids[:n_test])

    def extract(subset):
        X = np.array([[float(r[c]) for c in FEATURE_COLS] for r in subset], dtype=np.float32)
        y = np.array([[float(r[c]) for c in TARGET_COLS] for r in subset], dtype=np.float32)
        return X, y

    train_rows = [r for r in rows if int(r["run_id"]) not in test_ids]
    test_rows = [r for r in rows if int(r["run_id"]) in test_ids]

    return extract(train_rows), extract(test_rows), test_ids


print("=" * 60)
print("Phase A: Loading EnergyPlus simulation data")
print("=" * 60)

(X_train, y_train), (X_test, y_test), test_run_ids = load_data(DATA_PATH)

# Normalize inputs
x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0) + 1e-8
X_train_n = (X_train - x_mean) / x_std
X_test_n = (X_test - x_mean) / x_std

# Normalize targets for stable multi-output training
y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0) + 1e-8
y_train_n = (y_train - y_mean) / y_std
y_test_n = (y_test - y_mean) / y_std

print(f"Data:     {DATA_PATH.name}")
print(f"Training: {X_train.shape[0]} samples ({X_train.shape[0]//12} runs)")
print(f"Test:     {X_test.shape[0]} samples ({len(test_run_ids)} runs: {sorted(test_run_ids)})")
print(f"\nFeatures ({N_FEATURES}):")
for i, col in enumerate(FEATURE_COLS):
    print(f"  {col:<25s}  [{X_train[:,i].min():8.3f} – {X_train[:,i].max():8.3f}]")
print(f"\nTargets ({N_TARGETS}):")
for i, col in enumerate(TARGET_COLS):
    print(f"  {col:<25s}  [{y_train[:,i].min():8.1f} – {y_train[:,i].max():8.1f}]  mean={y_train[:,i].mean():.1f}")

# ── Phase B: Train MLP in PyTorch ─────────────────────────────────────────

class EnergyNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_out),
        )

    def forward(self, x):
        return self.net(x)


print(f"\n{'=' * 60}")
print("Phase B: Training PyTorch model")
print("=" * 60)

model = EnergyNet(N_FEATURES, N_TARGETS)
n_params = sum(p.numel() for p in model.parameters())
print(f"Architecture: {N_FEATURES} → 64 → 64 → 32 → {N_TARGETS}  ({n_params:,} parameters)")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
loss_fn = nn.MSELoss()

X_t = torch.from_numpy(X_train_n)
y_t = torch.from_numpy(y_train_n)
X_te = torch.from_numpy(X_test_n)
y_te = torch.from_numpy(y_test_n)

for epoch in range(1, 501):
    model.train()
    pred = model(X_t)
    loss = loss_fn(pred, y_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            test_pred_n = model(X_te).numpy()
            # Denormalize for RMSE in kWh
            test_pred = test_pred_n * y_std + y_mean
            rmse_elec = np.sqrt(np.mean((test_pred[:, 0] - y_test[:, 0]) ** 2))
            rmse_gas = np.sqrt(np.mean((test_pred[:, 1] - y_test[:, 1]) ** 2))
        print(f"  Epoch {epoch:3d}  loss={loss.item():.4f}  "
              f"RMSE elec={rmse_elec:.0f} kWh  gas={rmse_gas:.0f} kWh  "
              f"lr={optimizer.param_groups[0]['lr']:.1e}")

model.eval()
with torch.no_grad():
    y_pred_n = model(X_te).numpy()
    y_pred_pt = y_pred_n * y_std + y_mean

for i, col in enumerate(TARGET_COLS):
    rmse = np.sqrt(np.mean((y_pred_pt[:, i] - y_test[:, i]) ** 2))
    mean_y = y_test[:, i].mean()
    print(f"\n  {col}: RMSE={rmse:.1f} kWh  (mean={mean_y:.0f}, RMSE/mean={rmse/mean_y*100:.1f}%)")

# ── Phase C: Convert to CoreML ────────────────────────────────────────────

import coremltools as ct

print(f"\n{'=' * 60}")
print("Phase C: Converting to CoreML")
print("=" * 60)

# Wrap model to include denormalization so CoreML outputs real kWh
class EnergyNetExport(nn.Module):
    def __init__(self, net, y_mean, y_std):
        super().__init__()
        self.net = net
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32))

    def forward(self, x):
        return self.net(x) * self.y_std + self.y_mean

export_model = EnergyNetExport(model, y_mean, y_std)
export_model.eval()

example = torch.randn(1, N_FEATURES)
traced = torch.jit.trace(export_model, example)

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="calibration_params", shape=(1, N_FEATURES))],
    outputs=[ct.TensorType(name="energy_kwh")],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.macOS15,
)

model_path = str(Path(__file__).resolve().parent.parent / "models" / "energy_surrogate.mlpackage")
mlmodel.save(model_path)
print(f"Model saved to {model_path}")
print(f"Input:  {N_FEATURES} features (normalized): {FEATURE_COLS}")
print(f"Output: {N_TARGETS} targets (kWh): {TARGET_COLS}")

# ── Phase D: Run inference via CoreML ─────────────────────────────────────

print(f"\n{'=' * 60}")
print("Phase D: CoreML inference")
print("=" * 60)

cml = ct.models.MLModel(model_path)

# Compare predictions on full test set
y_pred_cml = []
for i in range(len(X_test_n)):
    out = cml.predict({"calibration_params": X_test_n[i:i+1]})
    y_pred_cml.append(out["energy_kwh"].flatten())
y_pred_cml = np.array(y_pred_cml)

max_err = np.max(np.abs(y_pred_cml - y_pred_pt))
print(f"Max error CoreML vs PyTorch: {max_err:.4f} kWh")

for i, col in enumerate(TARGET_COLS):
    cml_rmse = np.sqrt(np.mean((y_pred_cml[:, i] - y_test[:, i]) ** 2))
    print(f"CoreML {col} RMSE: {cml_rmse:.1f} kWh")

# Sample predictions
print(f"\n{'#':<3} {'WallIns':>7} {'Infil':>5} {'COP':>5} {'LPD':>5} {'Mon':>3} │ "
      f"{'Elec':>7} {'Pred':>7} {'Err':>6} │ {'Gas':>7} {'Pred':>7} {'Err':>6}")
print("─" * 82)
for i in range(min(15, len(X_test))):
    x = X_test[i]
    print(f"{i:<3} {x[0]:7.4f} {x[1]:5.2f} {x[2]:5.2f} {x[3]:5.1f} {x[4]:3.0f} │ "
          f"{y_test[i,0]:7.0f} {y_pred_cml[i,0]:7.0f} {y_pred_cml[i,0]-y_test[i,0]:+6.0f} │ "
          f"{y_test[i,1]:7.0f} {y_pred_cml[i,1]:7.0f} {y_pred_cml[i,1]-y_test[i,1]:+6.0f}")

# Benchmark
print(f"\nBenchmarking CoreML inference...")
sample = {"calibration_params": X_test_n[0:1]}
for _ in range(50):
    cml.predict(sample)

iters = 1000
t0 = time.perf_counter()
for _ in range(iters):
    cml.predict(sample)
elapsed = time.perf_counter() - t0

print(f"{iters} predictions in {elapsed*1000:.1f} ms  ({elapsed/iters*1000:.3f} ms/pred)")

# Save normalization stats for downstream use
stats_path = Path(__file__).resolve().parent.parent / "models" / "norm_stats.npz"
np.savez(stats_path, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
print(f"\nNormalization stats saved to {stats_path}")
