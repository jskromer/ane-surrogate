# ane-surrogate

ANE-accelerated surrogate model for building energy calibration. Runs parametric EnergyPlus simulations, trains a neural net on the results, and deploys via CoreML to Apple's Neural Engine — **30,000 predictions/sec** on Apple Silicon.

## Why a surrogate?

Bayesian calibration of building energy models requires thousands of forward simulations. EnergyPlus takes ~5 seconds per annual run. A surrogate model trained on 50 EnergyPlus runs replaces that with **0.035 ms per prediction** — a 140,000× speedup — making MCMC-based calibration practical on a laptop.

## Pipeline

```
EnergyPlus parametric batch  →  Training CSV  →  PyTorch MLP  →  CoreML  →  ANE inference
        (50 runs)              (600 rows)       (6,690 params)              (0.035 ms/pred)
```

## Training data

Generated from the **DOE Reference Small Office** (RefBldgSmallOfficeNew2004_Chicago.idf) — a 511 m², 5-zone building with PSZ-AC gas furnace HVAC, simulated against Chicago TMY3 weather.

**50 Latin Hypercube samples** across 4 calibration parameters, each producing 12 monthly energy totals (600 rows):

| Parameter | Range | Units | Baseline |
|---|---|---|---|
| Wall insulation thickness | 0.021 – 0.149 | m | 0.0495 |
| Infiltration multiplier | 0.53 – 2.49 | × | 1.0 |
| Cooling COP | 2.55 – 5.45 | W/W | 3.667 |
| Lighting power density | 5.09 – 19.91 | W/m² | 10.76 |

### Output ranges

| Target | Min | Max | Mean | Std |
|---|---|---|---|---|
| Electricity (kWh/month) | 3,021 | 10,067 | 6,125 | 1,704 |
| Natural gas (kWh/month) | 254 | 8,413 | 1,708 | 1,923 |
| Cooling elec. (kWh/month) | 0 | 2,081 | 416 | 531 |
| Heating gas (kWh/month) | 0 | 8,130 | 1,441 | 1,918 |
| Lighting elec. (kWh/month) | 564 | 2,621 | 1,523 | 534 |

Annual totals range from 41,500–106,300 kWh electricity and 10,600–35,100 kWh gas across the parameter space.

## Surrogate model performance

**Architecture**: 5 → 64 → 64 → 32 → 2 (6,690 parameters)

Trained on 40 runs (480 samples), tested on 10 held-out runs (120 samples):

| Target | RMSE | Mean | RMSE/Mean |
|---|---|---|---|
| Electricity | 359 kWh | 6,997 kWh | **5.1%** |
| Natural gas | 572 kWh | 1,784 kWh | 32.1% |

Gas RMSE is higher due to extreme seasonality (near-zero in summer, 8,000+ in winter). This improves with more training runs.

## Quick start

```bash
# Install dependencies
pip3 install torch coremltools numpy

# Generate training data (requires EnergyPlus 25.2 at /Applications/EnergyPlus-25-2-0/)
python3 scripts/run_eplus_batch.py --num-runs 50 --workers 4

# Train surrogate and deploy to CoreML
python3 scripts/energy_predictor.py
```

## Project structure

```
scripts/
  run_eplus_batch.py      Parametric EnergyPlus runner (LHS sampling, IDF modification, batch execution)
  energy_predictor.py     Train PyTorch MLP → convert to CoreML → benchmark on ANE
data/
  eplus_training_data.csv 50 runs × 12 months, 5 energy targets per row
models/
  energy_surrogate.mlpackage  CoreML model (denormalized outputs, ready for ANE)
  norm_stats.npz              Input normalization statistics (x_mean, x_std)
```

## Scaling up

The batch runner supports any number of runs and parallel workers:

```bash
# 500 runs for production-quality surrogate
python3 scripts/run_eplus_batch.py --num-runs 500 --workers 8

# Custom output location
python3 scripts/run_eplus_batch.py --num-runs 200 --output data/large_run.csv
```

At ~1.5 s/run with 4 workers, 500 runs takes ~3 minutes.

## MCP server

An MCP tool server exposes the surrogate to Claude (Desktop or Code), enabling instant energy predictions without Docker or EnergyPlus.

### Setup

```bash
# Install dependencies
uv sync

# Interactive testing with MCP inspector
uv run mcp dev mcp_server.py
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ane-surrogate": {
      "command": "uv",
      "args": [
        "--directory", "/Users/jskromer/Projects/ane-surrogate",
        "run", "mcp_server.py"
      ]
    }
  }
}
```

### Tools

| Tool | Description |
|---|---|
| `predict_energy` | Predict electricity and gas for one month |
| `compare_scenarios` | Compare 2-4 scenarios across all 12 months |
| `sweep_parameter` | Vary one parameter across its range |
| `get_parameter_info` | Return valid ranges and baselines |

Example prompts:
- "What energy would a building use with COP 4.0 and 15 W/m² lighting in January?"
- "Compare baseline vs high-efficiency (COP 5.0, 0.12 m insulation)"
- "Show me how electricity changes as I vary COP from 2.5 to 5.5"

## Requirements

- macOS 15+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- EnergyPlus 25.2.0 (at `/Applications/EnergyPlus-25-2-0/`) — only for generating training data
- PyTorch, coremltools, numpy
