# What If Your Building Energy Model Could Answer in Milliseconds?

## The Problem Every M&V Professional Knows

You're sitting with a building owner. They ask: "What if we upgrade the lighting *and* the HVAC — what's the combined savings?" You know the answer lives inside an EnergyPlus simulation, but running one takes minutes. Calibrating to their actual utility bills? That's dozens of runs. Exploring which combination of retrofit measures gives the best return? Hundreds.

So instead, you do what everyone does: you simplify. You use rules of thumb. You run one or two scenarios and call it a day. The physics-based model — the one that actually captures interactive effects between measures — sits on the shelf because it's too slow for a real conversation.

## What We Built

The ANE Surrogate is a neural network trained on real EnergyPlus simulation data. It learns the *relationships* between building parameters and energy consumption, then makes predictions 30,000 times per second — on a laptop.

Here's the pipeline:

1. Run EnergyPlus in batch mode with systematically varied parameters (wall insulation, infiltration, cooling efficiency, lighting power density)
2. Collect the monthly electricity and gas results
3. Train a small neural network to reproduce those results
4. Convert the model to run on Apple's Neural Engine — a dedicated AI chip that draws near-zero power

The result: a model that captures the physics of a calibrated EnergyPlus simulation but responds instantly.

## What Makes This Different

**It captures interactive effects.** When you install LEDs, you reduce lighting electricity — but you also remove waste heat the building was using for winter heating. Gas consumption goes *up*. Our surrogate reproduces this interaction because it learned from the full physics model, not from simplified engineering calculations. In our test case, an LED retrofit from 10.76 to 6.0 W/m² cut electricity by 29% but increased gas by 44%.

**It's an MCP server.** The surrogate exposes four tools that any AI assistant can call directly:

- *predict_energy* — give it building parameters and a month, get electricity and gas back instantly
- *compare_scenarios* — pit baseline against up to three ECM packages across all 12 months
- *sweep_parameter* — see how one variable affects consumption across its full range
- *get_parameter_info* — check valid ranges and baseline values

This means you can have a natural-language conversation about building retrofits and the AI assistant runs the physics model behind the scenes. No file management. No simulation queue. No post-processing.

**It runs on a laptop chip.** The Apple Neural Engine on an M4 Mac Mini delivers 30,000 predictions per second at near-zero power draw. No cloud. No GPU rental. No internet required. The model file is 12 KB.

## A Practical Example

We compared four scenarios for a 511 m² office building in Chicago:

| Scenario | Annual Electricity | Annual Gas | Elec. Savings | Gas Change |
|---|---|---|---|---|
| Baseline | 67,104 kWh | 20,213 kWh | — | — |
| LED Retrofit (6 W/m²) | 47,602 kWh | 29,071 kWh | -29% | +44% |
| HVAC Upgrade (COP 5.0) | 66,260 kWh | 20,624 kWh | -1.3% | +2% |
| All ECMs Combined | 43,382 kWh | 23,872 kWh | -35% | +18% |

The HVAC upgrade barely moves the needle as a standalone measure — the cooling load in a small Chicago office isn't dominant enough. But the combined package captures real interactions: the envelope and air sealing improvements partially offset the heating penalty from reduced lighting heat. These are exactly the kinds of insights that get lost when you analyze measures independently.

Total computation time for all four scenarios across all 12 months: under 2 milliseconds.

## Why This Matters for M&V

The central question in any retrofit savings claim is: what *would* the building have consumed if we hadn't intervened? That's the counterfactual. The quality of your counterfactual determines the quality of your savings estimate — everything else is accounting.

Building a competent counterfactual from a physics-based simulation requires calibration: adjusting model parameters until the simulation reproduces measured consumption. That means running the model hundreds of times to explore the parameter space. With EnergyPlus runs measured in minutes, calibration becomes a bottleneck that pushes practitioners toward shortcuts — single-point estimates, rules of thumb, or skipping calibration altogether.

A surrogate model that runs at 30,000 predictions per second removes that bottleneck. You can explore the full parameter space, run proper optimization loops, and deliver confidence intervals on savings — not just point estimates. You can ask "how sensitive is my savings claim to the infiltration rate I assumed?" and get an answer in milliseconds.

This is what makes a counterfactual *competent*: not that it passed a threshold test, but that you understand the uncertainty in your assumptions and can quantify how that uncertainty propagates into your savings estimate.

## What's Next

This is a proof of concept built on 50 EnergyPlus runs with 4 calibration parameters. The architecture scales to more complex buildings, more parameters, and hourly data. The immediate roadmap:

- Scale training data to 200–500 EnergyPlus runs for better gas prediction (currently 32% RMSE on gas vs. 5.1% on electricity)
- Build the calibration optimization loop — compare surrogate predictions to measured utility bills and find the best-fit parameter set
- Add hourly predictions for detailed M&V applications
- Connect to the "dual digital twin" workflow: physics-based simulation for the counterfactual, statistical model for the baseline

The code is open source at github.com/jskromer/ane-surrogate.

---

*Steve Kromer — Counterfactual Designs*
*Building Energy M&V | CMVP*
