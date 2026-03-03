"""
Building Energy Predictor: PyTorch → CoreML → ANE

Predicts hourly building energy consumption (kWh) from envelope, HVAC,
and weather parameters. Synthetic data follows EnergyPlus-style physics;
swap generate_data() with real E+ output later.
"""
import numpy as np
import torch
import torch.nn as nn
import time

# ── Feature definitions ────────────────────────────────────────────────

FEATURES = [
    # Envelope
    ("wall_r",      "Wall R-value",           "m²·K/W",  1.0,   8.0),
    ("roof_r",      "Roof R-value",           "m²·K/W",  2.0,  12.0),
    ("win_u",       "Window U-factor",        "W/m²·K",  0.8,   5.8),
    ("win_shgc",    "Window SHGC",            "",         0.2,   0.8),
    ("infil_ach",   "Infiltration ACH",       "1/h",      0.1,   1.5),
    # HVAC
    ("hvac_cop",    "HVAC COP",              "",          2.0,   6.0),
    ("hvac_cap",    "HVAC capacity",         "kW",       10.0,  80.0),
    ("setpoint_h",  "Heating setpoint",      "°C",       18.0,  22.0),
    ("setpoint_c",  "Cooling setpoint",      "°C",       23.0,  27.0),
    # Internal loads
    ("lpd",         "Lighting power density", "W/m²",     5.0,  20.0),
    ("epd",         "Equipment power density","W/m²",     5.0,  25.0),
    ("occupancy",   "Occupant density",       "ppl/100m²",1.0,  10.0),
    # Weather (hourly)
    ("temp_out",    "Outdoor dry-bulb temp",  "°C",      -15.0,  42.0),
    ("humidity_out", "Outdoor rel. humidity",  "%",        10.0,  100.0),
    ("solar_ghi",   "Global horiz. irradiance","W/m²",    0.0, 1000.0),
    ("wind_speed",  "Wind speed",             "m/s",       0.0,  15.0),
    # Time
    ("hour",        "Hour of day",            "h",         0.0,  23.0),
    ("month",       "Month of year",          "",          1.0,  12.0),
]

N_FEATURES = len(FEATURES)

# ── Phase A: Generate synthetic training data ──────────────────────────

def generate_data(n, seed=42):
    """
    Generate synthetic building energy data with EnergyPlus-style physics.
    Each sample is one hour of one building configuration + weather snapshot.
    """
    rng = np.random.RandomState(seed)

    # Sample each feature uniformly within its range
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    for i, (_, _, _, lo, hi) in enumerate(FEATURES):
        X[:, i] = rng.uniform(lo, hi, n).astype(np.float32)

    # Unpack for readability
    wall_r    = X[:, 0];  roof_r     = X[:, 1]
    win_u     = X[:, 2];  win_shgc   = X[:, 3]
    infil_ach = X[:, 4]
    hvac_cop  = X[:, 5];  hvac_cap   = X[:, 6]
    set_h     = X[:, 7];  set_c      = X[:, 8]
    lpd       = X[:, 9];  epd        = X[:,10]
    occ       = X[:,11]
    temp_out  = X[:,12];  hum_out    = X[:,13]
    solar     = X[:,14];  wind       = X[:,15]
    hour      = X[:,16];  month      = X[:,17]

    # Floor area (fixed for now — could vary later)
    area = 500.0  # m²

    # ── Envelope heat transfer (simplified UA method) ──
    wall_ua = area * 0.6 / wall_r          # walls ~60% of envelope area
    roof_ua = area * 0.25 / roof_r         # roof ~25%
    win_ua  = area * 0.15 * win_u          # windows ~15%
    infil_ua = area * 2.5 * 0.34 * infil_ach  # ρ·cp·V·ACH, ceiling=2.5m
    total_ua = wall_ua + roof_ua + win_ua + infil_ua

    # Temperature difference from setpoint
    # Heating when cold, cooling when hot, dead band in between
    temp_mid = (set_h + set_c) / 2.0
    delta_t = temp_out - temp_mid
    heating_load = np.maximum(0, set_h - temp_out) * total_ua / 1000  # kW
    cooling_load = np.maximum(0, temp_out - set_c) * total_ua / 1000  # kW

    # Solar gains through windows (reduce cooling / offset heating)
    solar_gain = area * 0.15 * win_shgc * solar / 1000  # kW
    cooling_load = np.maximum(0, cooling_load + solar_gain * 0.5)
    heating_load = np.maximum(0, heating_load - solar_gain * 0.3)

    # Wind increases infiltration losses
    wind_factor = 1.0 + 0.05 * wind
    heating_load *= wind_factor
    cooling_load *= wind_factor * 0.5

    # HVAC energy = thermal load / COP, capped by capacity
    hvac_thermal = np.minimum(heating_load + cooling_load, hvac_cap)
    hvac_energy = hvac_thermal / hvac_cop  # kW

    # ── Internal loads ──
    # Occupancy schedule: gaussian peak at 10am and 2pm on weekdays
    occ_schedule = (np.exp(-((hour - 10) ** 2) / 8) +
                    np.exp(-((hour - 14) ** 2) / 8))
    occ_schedule = np.clip(occ_schedule, 0.1, 1.0)

    lighting_kw = lpd * area / 1000 * occ_schedule
    equipment_kw = epd * area / 1000 * occ_schedule * 0.8
    body_heat_kw = occ * area / 100 * 0.12 * occ_schedule  # ~120W/person

    # Internal gains reduce heating, increase cooling
    internal_gain = lighting_kw + equipment_kw + body_heat_kw

    # ── Total hourly energy ──
    total_kwh = hvac_energy + lighting_kw + equipment_kw

    # Seasonal adjustment (heating months vs cooling months)
    season = np.where((month >= 11) | (month <= 3), 1.15, 1.0)  # winter bump
    total_kwh *= season

    # Add noise (measurement error, unmodeled effects)
    noise = rng.normal(0, total_kwh * 0.05 + 1.0, n).astype(np.float32)
    total_kwh = np.maximum(0, total_kwh + noise)

    return X, total_kwh.reshape(-1, 1).astype(np.float32)


print("=" * 60)
print("Phase A: Generating synthetic building energy data")
print("=" * 60)

X_train, y_train = generate_data(20000, seed=42)
X_test, y_test = generate_data(4000, seed=99)

# Normalize inputs for better training (save stats for inference)
x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0) + 1e-8
X_train_n = (X_train - x_mean) / x_std
X_test_n = (X_test - x_mean) / x_std

print(f"Training: {X_train.shape[0]:,} samples, {N_FEATURES} features")
print(f"Test:     {X_test.shape[0]:,} samples")
print(f"Target range: {y_train.min():.1f} – {y_train.max():.1f} kWh")
print(f"\nFeatures:")
for name, desc, unit, lo, hi in FEATURES:
    col = [i for i, f in enumerate(FEATURES) if f[0] == name][0]
    print(f"  {desc:<28s} [{lo:>7.1f} – {hi:>7.1f}] {unit}")

# ── Phase B: Train MLP in PyTorch ──────────────────────────────────────

class EnergyNet(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

print(f"\n{'=' * 60}")
print("Phase B: Training PyTorch model")
print("=" * 60)

model = EnergyNet(N_FEATURES)
n_params = sum(p.numel() for p in model.parameters())
print(f"Architecture: {N_FEATURES} → 128 → 64 → 32 → 1  ({n_params:,} parameters)")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
loss_fn = nn.MSELoss()

X_t = torch.from_numpy(X_train_n)
y_t = torch.from_numpy(y_train)
X_te = torch.from_numpy(X_test_n)
y_te = torch.from_numpy(y_test)

for epoch in range(1, 301):
    model.train()
    pred = model(X_t)
    loss = loss_fn(pred, y_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 50 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(X_te), y_te)
        rmse = np.sqrt(test_loss.item())
        print(f"  Epoch {epoch:3d}  train={loss.item():.2f}  test={test_loss.item():.2f}  RMSE={rmse:.2f} kWh  lr={optimizer.param_groups[0]['lr']:.1e}")

model.eval()
with torch.no_grad():
    y_pred_pt = model(X_te).numpy()

final_rmse = np.sqrt(np.mean((y_pred_pt - y_test) ** 2))
mean_y = y_test.mean()
print(f"\nFinal test RMSE: {final_rmse:.2f} kWh  (mean={mean_y:.1f}, RMSE/mean={final_rmse/mean_y*100:.1f}%)")

# ── Phase C: Convert to CoreML ─────────────────────────────────────────

import coremltools as ct

print(f"\n{'=' * 60}")
print("Phase C: Converting to CoreML")
print("=" * 60)

example = torch.randn(1, N_FEATURES)
traced = torch.jit.trace(model, example)

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="building_weather", shape=(1, N_FEATURES))],
    outputs=[ct.TensorType(name="energy_kwh")],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.macOS15,
)

model_path = "/tmp/energy_predictor.mlpackage"
mlmodel.save(model_path)
print(f"Model saved to {model_path}")
print(f"Input: {N_FEATURES} features (normalized), Output: energy_kwh")

# ── Phase D: Run inference via CoreML ──────────────────────────────────

print(f"\n{'=' * 60}")
print("Phase D: CoreML inference")
print("=" * 60)

cml = ct.models.MLModel(model_path)

# Compare predictions on full test set
y_pred_cml = []
for i in range(len(X_test_n)):
    out = cml.predict({"building_weather": X_test_n[i:i+1]})
    y_pred_cml.append(out["energy_kwh"].flatten()[0])
y_pred_cml = np.array(y_pred_cml).reshape(-1, 1)

max_err = np.max(np.abs(y_pred_cml - y_pred_pt))
cml_rmse = np.sqrt(np.mean((y_pred_cml - y_test) ** 2))
print(f"Max error CoreML vs PyTorch: {max_err:.4f} kWh")
print(f"CoreML test RMSE:            {cml_rmse:.2f} kWh")

# Sample predictions with building context
print(f"\n{'#':<4} {'WallR':>5} {'RoofR':>5} {'WinU':>5} {'COP':>4} {'Tout':>5} {'Solar':>5} {'Hour':>4} │ {'True':>7} {'Pred':>7} {'Err':>6}")
print("─" * 75)
for i in range(10):
    x = X_test[i]
    print(f"{i:<4} {x[0]:5.1f} {x[1]:5.1f} {x[2]:5.1f} {x[5]:4.1f} {x[12]:5.1f} {x[14]:5.0f} {x[16]:4.0f}  │ {y_test[i,0]:7.1f} {y_pred_cml[i,0]:7.1f} {y_pred_cml[i,0]-y_test[i,0]:+6.1f}")

# Benchmark
print(f"\nBenchmarking CoreML inference...")
sample = {"building_weather": X_test_n[0:1]}
for _ in range(50):
    cml.predict(sample)

iters = 1000
t0 = time.perf_counter()
for _ in range(iters):
    cml.predict(sample)
elapsed = time.perf_counter() - t0

print(f"{iters} predictions in {elapsed*1000:.1f} ms  ({elapsed/iters*1000:.3f} ms/pred)")
print(f"\nReady for real EnergyPlus data — replace generate_data() with CSV loader.")
