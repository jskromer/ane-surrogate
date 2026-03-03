#!/usr/bin/env python3
"""
EnergyPlus Batch Parametric Runner for Surrogate Calibration

Varies wall insulation, infiltration, HVAC COP, and lighting density across
Latin Hypercube samples, runs EnergyPlus simulations, and collects monthly
energy results into a single CSV for surrogate model training.

Base model: DOE Reference Small Office (RefBldgSmallOfficeNew2004_Chicago.idf)
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────

EPLUS_DIR = Path("/Applications/EnergyPlus-25-2-0")
EPLUS_BIN = EPLUS_DIR / "energyplus"
BASE_IDF = EPLUS_DIR / "ExampleFiles" / "RefBldgSmallOfficeNew2004_Chicago.idf"
WEATHER_FILE = (
    EPLUS_DIR / "WeatherData" / "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eplus_training_data.csv"

# ── Parameter ranges ──────────────────────────────────────────────────────

PARAMS = {
    "wall_insul_thickness": (0.02, 0.15),      # m
    "infil_multiplier":     (0.5, 2.5),         # ×baseline
    "cooling_cop":          (2.5, 5.5),          # W/W
    "lighting_density":     (5.0, 20.0),         # W/m²
}

# Baseline values in the IDF (used for infiltration scaling)
BASELINE_INFIL_ACH = 0.36         # Core_ZN: AirChanges/Hour
BASELINE_INFIL_FLOW = 0.000302    # Perimeter zones: Flow/ExteriorArea m³/s-m²
BASELINE_WALL_THICKNESS = 0.0495494599433393
BASELINE_COOLING_COP = 3.66668442928701
BASELINE_LPD = 10.76

# Meters to collect from eplusMtr.csv (EnergyPlus meter names)
METERS_OF_INTEREST = [
    "Electricity:Facility",
    "NaturalGas:Facility",
    "Cooling:Electricity",
    "Heating:NaturalGas",
    "InteriorLights:Electricity",
]


# ── Latin Hypercube Sampling (numpy-only) ─────────────────────────────────

def latin_hypercube(n_samples, n_dims, seed=42):
    """Generate LHS samples in [0, 1]^n_dims using numpy."""
    rng = np.random.RandomState(seed)
    samples = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            samples[perm[i], d] = (i + rng.uniform()) / n_samples
    return samples


def generate_samples(n_samples, seed=42):
    """Generate parameter samples scaled to physical ranges."""
    unit_samples = latin_hypercube(n_samples, len(PARAMS), seed=seed)
    param_names = list(PARAMS.keys())
    samples = []
    for i in range(n_samples):
        row = {}
        for j, name in enumerate(param_names):
            lo, hi = PARAMS[name]
            row[name] = lo + unit_samples[i, j] * (hi - lo)
        samples.append(row)
    return samples


# ── IDF Modification ──────────────────────────────────────────────────────

def modify_idf(idf_text, params):
    """Apply parameter values to the IDF text via regex replacement."""

    # 1. Wall insulation thickness
    #    Material object "Mass NonRes Wall Insulation" — replace Thickness field
    idf_text = re.sub(
        r"(Mass NonRes Wall Insulation,\s*!-\s*Name\s*\n"
        r"\s*MediumRough,\s*!-\s*Roughness\s*\n"
        r"\s*)[\d.]+(\s*,\s*!-\s*Thickness\s*\{m\})",
        rf"\g<1>{params['wall_insul_thickness']:.6f}\2",
        idf_text,
    )

    # 2. Infiltration — scale all zone infiltration rates
    mult = params["infil_multiplier"]

    # Core_ZN: AirChanges/Hour field
    new_ach = BASELINE_INFIL_ACH * mult
    idf_text = re.sub(
        r"(Core_ZN_Infiltration.*?Air Changes per Hour\s*\{1/hr\}\s*\n\s*)"
        r"[\d.]+(\s*,\s*!-\s*Constant Term)",
        rf"\g<1>{new_ach:.4f}\2",
        idf_text,
        flags=re.DOTALL,
    )

    # Perimeter zones: Flow/ExteriorArea field
    new_flow = BASELINE_INFIL_FLOW * mult
    idf_text = re.sub(
        r"(Flow Rate per Exterior Surface Area\s*\{m3/s-m2\}\s*\n\s*)"
        r",(\s*!-\s*Air Changes per Hour)",
        rf"\g<1>{new_flow:.6f},\2",  # was empty, now has value
        idf_text,
    )
    # Actually the perimeter zones use Flow/ExteriorArea method, so the value
    # is on the "Flow Rate per Exterior Surface Area" line itself.
    idf_text = re.sub(
        r"(\s+)0\.000302(\s*,\s*!-\s*Flow Rate per Exterior Surface Area\s*\{m3/s-m2\})",
        rf"\g<1>{new_flow:.6f}\2",
        idf_text,
    )

    # 3. Cooling COP — all Coil:Cooling:DX:SingleSpeed objects
    idf_text = re.sub(
        r"3\.66668442928701(\s*,\s*!-\s*Gross Rated Cooling COP\s*\{W/W\})",
        rf"{params['cooling_cop']:.4f}\1",
        idf_text,
    )

    # 4. Lighting power density — all Lights objects (Watts per Floor Area)
    #    Match the value before "!- Watts per Floor Area {W/m2}" in Lights objects
    idf_text = re.sub(
        r"(\s+)10\.76(\s*,\s*!-\s*Watts per Floor Area\s*\{W/m2\})",
        rf"\g<1>{params['lighting_density']:.2f}\2",
        idf_text,
    )

    # 5. Enable annual weather-file simulation (base IDF only runs sizing)
    #    SimulationControl fields: value precedes comment on same line
    idf_text = re.sub(
        r"YES(,\s+!-\s+Run Simulation for Sizing Periods)",
        r"NO\1",
        idf_text,
    )
    idf_text = re.sub(
        r"NO(,\s+!-\s+Run Simulation for Weather File Run Periods)",
        r"YES\1",
        idf_text,
    )

    # 6. Change Output:Meter from HOURLY to MONTHLY
    idf_text = re.sub(
        r"(Output:Meter\s*,.+?,\s*)HOURLY",
        r"\1MONTHLY",
        idf_text,
        flags=re.IGNORECASE,
    )

    # 7. Strip Output:Variable lines (reduce output size)
    idf_text = re.sub(
        r"\s*Output:Variable\s*,.*?;\s*\n",
        "\n",
        idf_text,
    )

    return idf_text


# ── Run a single simulation ───────────────────────────────────────────────

def run_single(run_id, params, base_idf_text, weather_file, work_dir):
    """Run one EnergyPlus simulation and return parsed monthly meter data."""
    run_dir = Path(work_dir) / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write modified IDF
    modified_idf = modify_idf(base_idf_text, params)
    idf_path = run_dir / "in.idf"
    idf_path.write_text(modified_idf)

    # Run EnergyPlus
    cmd = [
        str(EPLUS_BIN),
        "-w", str(weather_file),
        "-d", str(run_dir),
        "-r",  # run ReadVarsESO to produce CSV
        str(idf_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min max per run
    )

    if result.returncode != 0:
        err_file = run_dir / "eplusout.err"
        err_msg = ""
        if err_file.exists():
            err_msg = err_file.read_text()[-500:]
        return {
            "run_id": run_id,
            "success": False,
            "error": f"E+ exit code {result.returncode}: {err_msg}",
            "rows": [],
        }

    # Parse meter CSV
    mtr_csv = run_dir / "eplusMtr.csv"
    if not mtr_csv.exists():
        # Try alternate name
        mtr_csv = run_dir / "eplustbl.csv"

    rows = parse_meter_csv(run_id, params, mtr_csv)
    return {"run_id": run_id, "success": True, "error": None, "rows": rows}


def parse_meter_csv(run_id, params, mtr_csv_path):
    """Parse eplusMtr.csv for monthly totals of meters of interest.

    The meter CSV mixes reporting frequencies (e.g. hourly water heater rows
    alongside monthly facility totals).  Monthly rows are identified by having
    a non-empty value in the Electricity:Facility column.
    """
    rows = []

    if not mtr_csv_path.exists():
        return rows

    with open(mtr_csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Find column indices for meters of interest
        # Headers look like: "Electricity:Facility [J](Monthly)"
        col_map = {}
        elec_col = None
        for meter_name in METERS_OF_INTEREST:
            for idx, col in enumerate(header):
                if meter_name in col and "Monthly" in col:
                    col_map[meter_name] = idx
                    if meter_name == "Electricity:Facility":
                        elec_col = idx
                    break

        if not col_map:
            for meter_name in METERS_OF_INTEREST:
                for idx, col in enumerate(header):
                    if meter_name in col:
                        col_map[meter_name] = idx
                        if meter_name == "Electricity:Facility" and elec_col is None:
                            elec_col = idx
                        break

        month = 0
        for row_data in reader:
            if not row_data or not row_data[0].strip():
                continue

            # Only process rows that have monthly facility data (not hourly
            # sub-meter rows which leave the monthly columns empty)
            if elec_col is not None:
                val = row_data[elec_col].strip() if elec_col < len(row_data) else ""
                if not val:
                    continue

            date_str = row_data[0].strip()
            if "Annual" in date_str or "Run" in date_str:
                continue

            month += 1
            if month > 12:
                break

            result_row = {
                "run_id": run_id,
                "wall_insul_thickness": params["wall_insul_thickness"],
                "infil_multiplier": params["infil_multiplier"],
                "cooling_cop": params["cooling_cop"],
                "lighting_density": params["lighting_density"],
                "month": month,
            }

            for meter_name, col_idx in col_map.items():
                try:
                    val_j = float(row_data[col_idx])
                    val_kwh = val_j / 3.6e6
                except (ValueError, IndexError):
                    val_kwh = float("nan")

                short_name = (
                    meter_name.replace("Electricity:Facility", "electricity_kwh")
                    .replace("NaturalGas:Facility", "gas_kwh")
                    .replace("Cooling:Electricity", "cooling_kwh")
                    .replace("Heating:NaturalGas", "heating_gas_kwh")
                    .replace("InteriorLights:Electricity", "lighting_kwh")
                )
                result_row[short_name] = round(val_kwh, 2)

            rows.append(result_row)

    return rows


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run parametric EnergyPlus simulations for surrogate training"
    )
    parser.add_argument(
        "--num-runs", type=int, default=50,
        help="Number of LHS samples (default: 50)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers (default: 4)",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for LHS (default: 42)",
    )
    parser.add_argument(
        "--keep-runs", action="store_true",
        help="Keep individual run directories (default: clean up)",
    )
    parser.add_argument(
        "--work-dir", type=str, default=None,
        help="Working directory for E+ runs (default: system temp)",
    )
    args = parser.parse_args()

    # Validate paths
    if not EPLUS_BIN.exists():
        print(f"ERROR: EnergyPlus not found at {EPLUS_BIN}", file=sys.stderr)
        sys.exit(1)
    if not BASE_IDF.exists():
        print(f"ERROR: Base IDF not found at {BASE_IDF}", file=sys.stderr)
        sys.exit(1)
    if not WEATHER_FILE.exists():
        print(f"ERROR: Weather file not found at {WEATHER_FILE}", file=sys.stderr)
        sys.exit(1)

    # Read base IDF
    base_idf_text = BASE_IDF.read_text()

    # Generate parameter samples
    samples = generate_samples(args.num_runs, seed=args.seed)

    print("=" * 65)
    print("EnergyPlus Batch Parametric Runner")
    print("=" * 65)
    print(f"Base IDF:    {BASE_IDF.name}")
    print(f"Weather:     {WEATHER_FILE.name}")
    print(f"Runs:        {args.num_runs}")
    print(f"Workers:     {args.workers}")
    print(f"Output:      {args.output}")
    print(f"\nParameter ranges:")
    for name, (lo, hi) in PARAMS.items():
        print(f"  {name:25s}  [{lo:8.4f} – {hi:8.4f}]")
    print()

    # Set up working directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work_dir = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="eplus_batch_"))
        cleanup_work_dir = not args.keep_runs

    print(f"Work dir:    {work_dir}")
    print()

    # Run simulations
    all_rows = []
    failed = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for i, params in enumerate(samples):
            future = executor.submit(
                run_single, i, params, base_idf_text,
                str(WEATHER_FILE), str(work_dir),
            )
            futures[future] = i

        for future in as_completed(futures):
            run_id = futures[future]
            try:
                result = future.result()
            except Exception as e:
                failed.append((run_id, str(e)))
                print(f"  [FAIL] Run {run_id:3d}: {e}")
                continue

            if result["success"]:
                all_rows.extend(result["rows"])
                n_months = len(result["rows"])
                elapsed = time.time() - t_start
                done = len(all_rows) // 12 if all_rows else 0
                print(
                    f"  [OK]   Run {run_id:3d}  "
                    f"({n_months} months)  "
                    f"[{done}/{args.num_runs} done, {elapsed:.0f}s elapsed]"
                )
            else:
                failed.append((run_id, result["error"]))
                print(f"  [FAIL] Run {run_id:3d}: {result['error'][:80]}")

    elapsed_total = time.time() - t_start

    # Write output CSV
    if all_rows:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "run_id", "wall_insul_thickness", "infil_multiplier",
            "cooling_cop", "lighting_density", "month",
            "electricity_kwh", "gas_kwh", "cooling_kwh",
            "heating_gas_kwh", "lighting_kwh",
        ]
        # Sort by run_id then month
        all_rows.sort(key=lambda r: (r["run_id"], r["month"]))

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"\n{'=' * 65}")
        print(f"Results written to {output_path}")
        print(f"  Rows:     {len(all_rows)}")
        print(f"  Runs OK:  {len(all_rows) // 12}")
        print(f"  Failed:   {len(failed)}")
        print(f"  Time:     {elapsed_total:.1f}s ({elapsed_total/args.num_runs:.1f}s/run)")

        # Quick summary stats
        elec = [r["electricity_kwh"] for r in all_rows if "electricity_kwh" in r]
        gas = [r["gas_kwh"] for r in all_rows if "gas_kwh" in r]
        if elec:
            print(f"\n  Electricity (monthly kWh):  min={min(elec):.0f}  max={max(elec):.0f}  mean={np.mean(elec):.0f}")
        if gas:
            print(f"  Natural gas (monthly kWh): min={min(gas):.0f}  max={max(gas):.0f}  mean={np.mean(gas):.0f}")
    else:
        print("\nNo results collected. Check errors above.")

    if failed:
        print(f"\nFailed runs: {[r for r, _ in failed]}")

    # Cleanup
    if cleanup_work_dir and work_dir.exists():
        print(f"\nCleaning up {work_dir}...")
        shutil.rmtree(work_dir, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()
