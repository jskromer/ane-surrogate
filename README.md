# ane-surrogate

ANE-accelerated surrogate model for building energy prediction. Trains a neural net on building envelope, HVAC, and weather parameters to predict hourly energy consumption, then runs inference on Apple's Neural Engine via CoreML.

## Overview

Traditional building energy simulation (EnergyPlus) takes minutes per run. This project trains a surrogate model that approximates the same physics in **0.033 ms** — enabling real-time design exploration, optimization loops, and edge deployment on Apple Silicon.

## Pipeline

```
EnergyPlus data (or synthetic) → PyTorch MLP → CoreML → ANE inference
```

### Input features (18)

| Category | Features |
|----------|----------|
| Envelope | Wall R-value, Roof R-value, Window U-factor, Window SHGC, Infiltration ACH |
| HVAC | COP, Capacity, Heating setpoint, Cooling setpoint |
| Internal loads | Lighting power density, Equipment power density, Occupant density |
| Weather | Outdoor temp, Humidity, Solar GHI, Wind speed |
| Time | Hour of day, Month of year |

### Output

Hourly building energy consumption (kWh)

## Quick start

```bash
pip3 install torch coremltools numpy
python3 scripts/energy_predictor.py
```

## Project structure

```
data/       EnergyPlus CSV or synthetic training data
models/     Saved .pt checkpoints and .mlpackage CoreML models
scripts/    Training, conversion, and inference pipelines
```

## Status

Currently uses synthetic data with simplified physics. Next step: connect real EnergyPlus hourly output as training data.

## Requirements

- macOS 15+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- PyTorch, coremltools, numpy
