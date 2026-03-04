# ane-surrogate

Neural Engine-accelerated surrogate model for building energy calibration.

## Project structure

```
scripts/    Training, simulation, conversion, and inference scripts
data/       Training/test data (CSV from EnergyPlus batch runs)
models/     Saved PyTorch checkpoints and CoreML .mlpackage files
configs/    Parameter configs for batch runs
notebooks/  Analysis notebooks
tests/      Tests
```

## Key files

- `scripts/energy_predictor.py` — Synthetic data pipeline: generate data, train PyTorch MLP, convert to CoreML, run on ANE
- `scripts/run_eplus_batch.py` — Parametric EnergyPlus runner: LHS sampling, IDF modification, batch simulation, CSV output

## EnergyPlus setup

- **Installation**: `/Applications/EnergyPlus-25-2-0/`
- **Binary**: `/Applications/EnergyPlus-25-2-0/energyplus`
- **Base IDF**: `RefBldgSmallOfficeNew2004_Chicago.idf` (DOE Reference Small Office, 511 m², 5 zones)
- **Weather**: `USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw`

## Calibration parameters

| Parameter | Range | Units | Baseline |
|---|---|---|---|
| Wall insulation thickness | 0.02–0.15 | m | 0.0495 |
| Infiltration multiplier | 0.5–2.5 | × | 1.0 |
| Cooling COP | 2.5–5.5 | W/W | 3.667 |
| Lighting power density | 5.0–20.0 | W/m² | 10.76 |

## Running

```bash
# Quick test (2 runs, serial)
python3 scripts/run_eplus_batch.py --num-runs 2 --workers 1

# Full batch (50 runs, 4 parallel workers)
python3 scripts/run_eplus_batch.py --num-runs 50 --workers 4

# Output: data/eplus_training_data.csv (N_runs × 12 months)
```

## MCP server

`mcp_server.py` exposes the CoreML surrogate as an MCP tool server (FastMCP, stdio transport). No Docker or EnergyPlus required — runs natively on Apple Silicon.

### Setup

```bash
# Install dependencies (uses uv with pyproject.toml)
uv sync

# Start server (waits on stdin for MCP messages)
uv run mcp_server.py

# Interactive testing with MCP inspector
uv run mcp dev mcp_server.py
```

### Claude Desktop config

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

- `predict_energy` — single month prediction (all 4 params + month required)
- `compare_scenarios` — compare 2-4 named scenarios across all 12 months
- `sweep_parameter` — vary one parameter across its range, all 12 months per point
- `get_parameter_info` — parameter ranges and baselines

### Key details

- Inputs are physical units; server normalizes internally
- Outputs are total facility electricity and natural gas (kWh) — not sub-metered
- Logs to stderr (stdout is MCP stdio transport)

## Conventions

- Python 3.10+, PyTorch, coremltools, numpy
- Models output to `/tmp/` during development; production models go in `models/`
- Input features normalized (zero mean, unit variance) before inference
- CoreML models use `ComputeUnit.ALL` to enable ANE execution
- EnergyPlus outputs in Joules; scripts convert to kWh
- Monthly granularity for calibration training data
