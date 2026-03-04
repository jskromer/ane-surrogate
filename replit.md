# ane-surrogate

ANE-accelerated surrogate model for building energy calibration, adapted to run on Replit (Linux).

## Project Overview

A building energy surrogate model project with two components:

1. **Frontend site** (`src/App.jsx`) — React + Recharts interactive showcase of the surrogate model, with scenario comparison charts, pipeline visualization, and insight callouts. Served via Vite on port 5000.
2. **MCP server** (`mcp_server.py`) — FastMCP stdio server exposing 4 prediction tools to Claude/MCP clients. Uses numpy MLP inference.

## Frontend (React + Vite)

- **Entry**: `index.html` → `src/main.jsx` → `src/App.jsx`
- **Original source**: `docs/site/ane-surrogate-site.jsx` (from GitHub repo)
- **Dependencies**: react, react-dom, recharts, vite, @vitejs/plugin-react
- **Port**: 5000 (Vite dev server, 0.0.0.0)
- **Deploy**: Static site, `npx vite build` → `dist/`

## Backend (MCP Server)

- **`mcp_server.py`** — MCP stdio server exposing 4 tools to Claude
- **`models/norm_stats.npz`** — Input/output normalization statistics
- **`models/numpy_weights.npz`** — Trained numpy MLP weights (5→64→64→32→2)

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

1. Removed `coremltools` from `pyproject.toml` (Apple Silicon only)
2. Replaced CoreML inference with pure numpy forward pass in `mcp_server.py`
3. Trained numpy weights using physics-based synthetic data scaled to match original normalization statistics
4. Added React frontend from the repo's `docs/site/` JSX file

## Dependencies

**Python**: mcp[cli], numpy, flask  
**Node**: react, react-dom, recharts, vite, @vitejs/plugin-react
