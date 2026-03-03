# ane-surrogate

Neural Engine-accelerated surrogate model for building energy simulation.

## Project structure

```
data/       Training/test data (CSV from EnergyPlus or synthetic)
models/     Saved PyTorch checkpoints and CoreML .mlpackage files
scripts/    Training, conversion, and inference scripts
```

## Key files

- `scripts/energy_predictor.py` — End-to-end pipeline: generate data → train PyTorch MLP → convert to CoreML → run inference on ANE

## Conventions

- Python 3.9+, PyTorch, coremltools, numpy
- Models output to `/tmp/` during development; production models go in `models/`
- Input features are normalized (zero mean, unit variance) before inference
- CoreML models use `ComputeUnit.ALL` to enable ANE execution
- The 18 input features match EnergyPlus variable naming for future CSV integration

## Running

```bash
cd scripts
python3 energy_predictor.py
```

## Dependencies

```bash
pip3 install torch coremltools numpy
```
