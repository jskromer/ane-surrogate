"""
ANE Surrogate MCP Server

Exposes a CoreML building-energy surrogate model to Claude via MCP.
Predicts monthly electricity and natural gas consumption for a DOE Reference
Small Office building given 4 calibration parameters + month.

Runs on Apple's Neural Engine — 0.035 ms per prediction, no Docker required.
"""

import sys
import logging
from pathlib import Path

import coremltools as ct
import numpy as np
from mcp.server.fastmcp import FastMCP

# ── Logging (stderr only — stdout is MCP stdio transport) ────────────────

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ane-surrogate")

# ── Load CoreML model and normalization stats ────────────────────────────

MODEL_PATH = Path(__file__).parent / "models" / "energy_surrogate.mlpackage"
STATS_PATH = Path(__file__).parent / "models" / "norm_stats.npz"

log.info("Loading CoreML model from %s", MODEL_PATH)
model = ct.models.MLModel(str(MODEL_PATH))
stats = np.load(str(STATS_PATH))
x_mean, x_std = stats["x_mean"], stats["x_std"]
log.info("Model loaded — ready for predictions")

# ── Parameter metadata ───────────────────────────────────────────────────

PARAMS = {
    "wall_insul_thickness": {
        "description": "Wall insulation thickness",
        "units": "m",
        "min": 0.02,
        "max": 0.15,
        "baseline": 0.0495,
        "index": 0,
    },
    "infil_multiplier": {
        "description": "Infiltration rate multiplier",
        "units": "\u00d7",
        "min": 0.5,
        "max": 2.5,
        "baseline": 1.0,
        "index": 1,
    },
    "cooling_cop": {
        "description": "Cooling COP",
        "units": "W/W",
        "min": 2.5,
        "max": 5.5,
        "baseline": 3.667,
        "index": 2,
    },
    "lighting_density": {
        "description": "Lighting power density",
        "units": "W/m\u00b2",
        "min": 5.0,
        "max": 20.0,
        "baseline": 10.76,
        "index": 3,
    },
}

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

BASELINE = {name: p["baseline"] for name, p in PARAMS.items()}

# ── Inference helper ─────────────────────────────────────────────────────


def _predict(wall_insul_thickness: float, infil_multiplier: float,
             cooling_cop: float, lighting_density: float,
             month: int) -> tuple[float, float]:
    """Run a single prediction through the CoreML model.

    Inputs are in physical units. Returns (electricity_kwh, gas_kwh).
    """
    raw = np.array([[wall_insul_thickness, infil_multiplier,
                     cooling_cop, lighting_density, float(month)]],
                   dtype=np.float32)
    normalized = (raw - x_mean) / x_std
    result = model.predict({"calibration_params": normalized})
    out = result["energy_kwh"].flatten()
    return float(out[0]), float(out[1])


def _predict_all_months(params: dict) -> list[tuple[float, float]]:
    """Predict all 12 months for a parameter set. Returns list of (elec, gas)."""
    return [_predict(**params, month=m) for m in range(1, 13)]


def _validate_params(params: dict) -> str | None:
    """Validate parameter values are within range. Returns error message or None."""
    for name, value in params.items():
        if name not in PARAMS:
            return f"Unknown parameter: {name}. Valid: {', '.join(PARAMS.keys())}"
        info = PARAMS[name]
        if not (info["min"] <= value <= info["max"]):
            return (f"{name} = {value} is out of range "
                    f"[{info['min']}, {info['max']}]")
    return None


# ── MCP Server ───────────────────────────────────────────────────────────

mcp = FastMCP("ane-surrogate")


@mcp.tool()
def predict_energy(
    wall_insul_thickness: float,
    infil_multiplier: float,
    cooling_cop: float,
    lighting_density: float,
    month: int,
) -> str:
    """Predict monthly building energy consumption using the ANE surrogate model.

    Predicts total facility electricity and total natural gas for a DOE Reference
    Small Office (511 m², Chicago) given 4 calibration parameters and a month.

    These are whole-building totals, not sub-metered by end use.

    Args:
        wall_insul_thickness: Wall insulation thickness in meters (0.02 - 0.15, baseline 0.0495)
        infil_multiplier: Infiltration rate multiplier (0.5 - 2.5, baseline 1.0)
        cooling_cop: Cooling coefficient of performance in W/W (2.5 - 5.5, baseline 3.667)
        lighting_density: Lighting power density in W/m² (5.0 - 20.0, baseline 10.76)
        month: Month of year (1 = January, 12 = December)
    """
    if not (1 <= month <= 12):
        return f"Error: month must be 1-12, got {month}"

    params = {
        "wall_insul_thickness": wall_insul_thickness,
        "infil_multiplier": infil_multiplier,
        "cooling_cop": cooling_cop,
        "lighting_density": lighting_density,
    }
    err = _validate_params(params)
    if err:
        return f"Error: {err}"

    elec, gas = _predict(**params, month=month)

    lines = [
        f"Prediction for {MONTH_NAMES[month - 1]}:",
        f"  Electricity: {elec:,.0f} kWh",
        f"  Natural gas: {gas:,.0f} kWh",
        "",
        "Parameters used:",
        f"  Wall insulation: {wall_insul_thickness} m",
        f"  Infiltration multiplier: {infil_multiplier}\u00d7",
        f"  Cooling COP: {cooling_cop} W/W",
        f"  Lighting density: {lighting_density} W/m\u00b2",
        "",
        "Note: Outputs are total facility electricity and natural gas "
        "for a 511 m\u00b2 DOE Reference Small Office in Chicago, "
        "not sub-metered by end use.",
    ]
    return "\n".join(lines)


@mcp.tool()
def compare_scenarios(
    scenarios: list[dict],
) -> str:
    """Compare 2-4 building scenarios across all 12 months.

    Each scenario is a dict with 'name' (str) and 'params' (dict of parameter
    overrides). Any parameter not specified defaults to the baseline value.

    Example input:
        [
            {"name": "Baseline", "params": {}},
            {"name": "High efficiency", "params": {"cooling_cop": 5.0, "wall_insul_thickness": 0.12}}
        ]

    Args:
        scenarios: List of 2-4 scenario dicts, each with 'name' and 'params' keys.
    """
    if not (2 <= len(scenarios) <= 4):
        return f"Error: provide 2-4 scenarios, got {len(scenarios)}"

    # Validate all scenarios
    resolved = []
    for s in scenarios:
        name = s.get("name", "Unnamed")
        raw_params = s.get("params", {})
        params = {**BASELINE, **raw_params}
        err = _validate_params(params)
        if err:
            return f"Error in scenario '{name}': {err}"
        resolved.append((name, params))

    # Run predictions
    results = {}
    for name, params in resolved:
        monthly = _predict_all_months(params)
        results[name] = monthly

    # Format annual summary
    lines = ["Annual Summary", "=" * 60]
    header = f"{'Scenario':<25s} {'Electricity':>12s} {'Natural Gas':>12s}"
    lines.append(header)
    lines.append("-" * 60)
    for name, monthly in results.items():
        total_elec = sum(e for e, _ in monthly)
        total_gas = sum(g for _, g in monthly)
        lines.append(f"{name:<25s} {total_elec:>10,.0f} kWh {total_gas:>10,.0f} kWh")

    # Format monthly breakdown
    lines.append("")
    lines.append("Monthly Breakdown (kWh)")
    lines.append("=" * 60)

    for name, monthly in results.items():
        lines.append("")
        lines.append(f"--- {name} ---")
        lines.append(f"  {'Month':<5s} {'Electricity':>12s} {'Natural Gas':>12s}")
        for m, (elec, gas) in enumerate(monthly):
            lines.append(f"  {MONTH_NAMES[m]:<5s} {elec:>12,.0f} {gas:>12,.0f}")
        total_elec = sum(e for e, _ in monthly)
        total_gas = sum(g for _, g in monthly)
        lines.append(f"  {'Total':<5s} {total_elec:>12,.0f} {total_gas:>12,.0f}")

    return "\n".join(lines)


@mcp.tool()
def sweep_parameter(
    param_name: str,
    num_points: int = 10,
    other_params: dict | None = None,
) -> str:
    """Sweep one parameter across its full range and show energy for all 12 months.

    Varies the named parameter from its minimum to maximum in equal steps,
    holding all other parameters at baseline (or at values from other_params).

    Args:
        param_name: Parameter to sweep. One of: wall_insul_thickness, infil_multiplier, cooling_cop, lighting_density.
        num_points: Number of sweep points (default 10, max 20).
        other_params: Optional dict of non-swept parameters to override baseline values.
    """
    if param_name not in PARAMS:
        return f"Error: unknown parameter '{param_name}'. Valid: {', '.join(PARAMS.keys())}"

    num_points = max(2, min(num_points, 20))

    info = PARAMS[param_name]
    sweep_values = np.linspace(info["min"], info["max"], num_points)

    base = {**BASELINE}
    if other_params:
        err = _validate_params(other_params)
        if err:
            return f"Error in other_params: {err}"
        base.update(other_params)

    # Header
    lines = [
        f"Sweeping {info['description']} ({param_name})",
        f"  Range: {info['min']} - {info['max']} {info['units']}",
        f"  Points: {num_points}",
    ]
    if other_params:
        lines.append("  Overrides: " + ", ".join(
            f"{k}={v}" for k, v in other_params.items()))
    lines.append("")

    # Table header
    col_w = 8
    header = f"  {'Value':>{col_w}s}"
    for mn in MONTH_NAMES:
        header += f"  {mn:>{col_w}s}"
    header += f"  {'Annual':>{col_w}s}"

    # Electricity table
    lines.append("Electricity (kWh)")
    lines.append(header)
    lines.append("  " + "-" * (col_w + (col_w + 2) * 13))
    for val in sweep_values:
        params = {**base, param_name: float(val)}
        monthly = _predict_all_months(params)
        row = f"  {val:>{col_w}.4f}"
        for elec, _ in monthly:
            row += f"  {elec:>{col_w},.0f}"
        annual = sum(e for e, _ in monthly)
        row += f"  {annual:>{col_w},.0f}"
        lines.append(row)

    lines.append("")

    # Gas table
    lines.append("Natural Gas (kWh)")
    lines.append(header)
    lines.append("  " + "-" * (col_w + (col_w + 2) * 13))
    for val in sweep_values:
        params = {**base, param_name: float(val)}
        monthly = _predict_all_months(params)
        row = f"  {val:>{col_w}.4f}"
        for _, gas in monthly:
            row += f"  {gas:>{col_w},.0f}"
        annual = sum(g for _, g in monthly)
        row += f"  {annual:>{col_w},.0f}"
        lines.append(row)

    return "\n".join(lines)


@mcp.tool()
def get_parameter_info() -> str:
    """Return valid ranges and baseline values for all building parameters.

    Use this to understand what inputs the surrogate model accepts before
    calling predict_energy, compare_scenarios, or sweep_parameter.
    """
    lines = [
        "ANE Surrogate Model — Parameter Reference",
        "=" * 55,
        "",
        "Building: DOE Reference Small Office (511 m², 5-zone)",
        "Location: Chicago, IL (TMY3 weather)",
        "Model: 5 → 64 → 64 → 32 → 2 MLP on Apple Neural Engine",
        "Outputs: Total facility electricity and natural gas (kWh/month)",
        "",
        f"{'Parameter':<25s} {'Range':>15s} {'Units':>6s} {'Baseline':>10s}",
        "-" * 55,
    ]
    for name, info in PARAMS.items():
        range_str = f"{info['min']} - {info['max']}"
        lines.append(
            f"{name:<25s} {range_str:>15s} {info['units']:>6s} {info['baseline']:>10.4f}"
        )
    lines.extend([
        "",
        f"{'month':<25s} {'1 - 12':>15s} {'':>6s} {'—':>10s}",
        "",
        "Notes:",
        "- All parameters are required for predict_energy",
        "- compare_scenarios and sweep_parameter default to baseline values",
        "- Outputs are whole-building totals, not sub-metered by end use",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
