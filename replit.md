# ane-surrogate

ANE-accelerated surrogate model for building energy calibration, adapted to run on Replit (Linux).

## Project Overview

Exposes a building energy surrogate model as an MCP (Model Context Protocol) tool server. Predicts monthly electricity and natural gas consumption for a DOE Reference Small Office building (511 m², Chicago) given 4 calibration parameters and a month.

**Original design**: Apple Silicon / CoreML inference on Apple Neural Engine  
**Replit adaptation**: Pure numpy MLP inference (same architecture: 5→64→64→32→2)

## Architecture

- **`mcp_server.py`** — MCP stdio server exposing 4 tools to Claude
- **`models/norm_stats.npz`** — Input/output normalization statistics (x_mean, x_std, y_mean, y_std)
- **`models/numpy_weights.npz`** — Trained numpy MLP weights (W1/b1..W4/b4), 5→64→64→32→2
- **`scripts/energy_predictor.py`** — Original PyTorch → CoreML training pipeline (Apple only)
- **`scripts/run_eplus_batch.py`** — Parametric EnergyPlus batch runner (Apple only, requires EnergyPlus)

## MCP Tools

| Tool | Description |
|------|-------------|
| `predict_energy` | Single month prediction (all 4 params + month required) |
| `compare_scenarios` | Compare 2-4 named scenarios across all 12 months |
| `sweep_parameter` | Vary one parameter across its range, all 12 months per point |
| `get_parameter_info` | Parameter ranges and baselines |

## Calibration Parameters

| Parameter | Range | Units | Baseline |
|-----------|-------|-------|----------|
| wall_insul_thickness | 0.02–0.15 | m | 0.0495 |
| infil_multiplier | 0.5–2.5 | × | 1.0 |
| cooling_cop | 2.5–5.5 | W/W | 3.667 |
| lighting_density | 5.0–20.0 | W/m² | 10.76 |

## Replit Adaptations

1. **Removed `coremltools`** from `pyproject.toml` (Apple Silicon only)
2. **Replaced CoreML inference** with pure numpy forward pass in `mcp_server.py`
3. **Trained numpy weights** (`models/numpy_weights.npz`) using physics-based synthetic data scaled to match original normalization statistics
4. **Workflow**: Console workflow running `python3 mcp_server.py` (MCP stdio transport)

## Usage

The server uses MCP stdio transport — it reads JSON-RPC messages from stdin and writes responses to stdout. Logs go to stderr.

### Connect from Claude Desktop (on Apple machine)

```json
{
  "mcpServers": {
    "ane-surrogate": {
      "command": "python3",
      "args": ["mcp_server.py"]
    }
  }
}
```

## Dependencies

- `mcp[cli]` — FastMCP server framework
- `numpy` — Array math and MLP inference
