import { useState, useEffect, useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from "recharts";

const COPPER = "#B07D4F";
const CHARCOAL = "#2D2D2D";
const DARK_BG = "#FAFAF8";
const CARD_BG = "#FFFFFF";
const WARM_WHITE = "#1A1A1A";
const MUTED = "#6B6B6B";
const GREEN = "#2D8A4E";
const RED = "#C0392B";
const BLUE = "#2B6CB0";
const TEAL = "#1A8A8A";
const BORDER = "#E2E0DC";

const monthNames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

const scenarios = {
  baseline: {
    label: "Baseline",
    desc: "Existing building, no changes",
    params: "Wall: 0.05m | Infil: 1.0× | COP: 3.67 | LPD: 10.76 W/m²",
    elec: [5340,5256,5264,5420,5680,5952,6256,6164,5836,5420,5288,5228],
    gas: [5176,3388,1884,830,414,242,155,0,177,670,2424,4880],
    color: MUTED,
  },
  led: {
    label: "LED Retrofit",
    desc: "Lighting power density reduced to 6.0 W/m²",
    params: "LPD: 10.76 → 6.0 W/m²",
    elec: [3810,3712,3718,3746,3952,4228,4376,4340,4104,3876,3848,3892],
    gas: [5884,4136,2648,1465,880,557,436,528,828,1714,3760,6236],
    color: GREEN,
  },
  hvac: {
    label: "HVAC Upgrade",
    desc: "Cooling COP improved to 5.0 W/W",
    params: "COP: 3.67 → 5.0 W/W",
    elec: [5296,5204,5188,5284,5536,5812,6080,5992,5768,5484,5328,5288],
    gas: [5284,3508,2132,972,365,80,14,36,396,902,2240,4696],
    color: BLUE,
  },
  envelope: {
    label: "Envelope + Sealing",
    desc: "Better insulation and reduced air infiltration",
    params: "Wall: 0.05 → 0.12m | Infil: 1.0 → 0.6×",
    elec: [5072,5124,5248,5428,5624,5784,5936,5716,5460,5224,5056,5104],
    gas: [4224,2760,1669,746,286,85,14,82,227,458,1490,3668],
    color: TEAL,
  },
  all: {
    label: "All ECMs Combined",
    desc: "LED + HVAC + Envelope + Air Sealing",
    params: "LPD: 6.0 | COP: 5.0 | Wall: 0.12m | Infil: 0.6×",
    elec: [3684,3526,3396,3392,3624,3774,3924,3852,3718,3530,3422,3540],
    gas: [5192,3888,2616,1363,734,420,205,271,508,977,2562,5136],
    color: COPPER,
  },
};

const pipelineSteps = [
  { icon: "⚡", title: "EnergyPlus", time: "~3 min/run", detail: "50 batch simulations with Latin Hypercube sampling across 4 calibration parameters" },
  { icon: "📊", title: "Collect Data", time: "600 rows", detail: "Monthly electricity and gas from each simulation, plus weather and parameter values" },
  { icon: "🧮", title: "PyTorch", time: "~10 sec", detail: "Train 5→64→64→32→2 MLP that learns the physics relationships" },
  { icon: "🍎", title: "CoreML", time: "12 KB model", detail: "Convert trained model to Apple's ML format for hardware acceleration" },
  { icon: "🧠", title: "Apple Neural Engine", time: "0.033 ms", detail: "30,000 predictions/sec at near-zero power on the M4 chip" },
  { icon: "🔌", title: "MCP Server", time: "4 tools", detail: "predict_energy · compare_scenarios · sweep_parameter · get_parameter_info" },
];

const fmt = (n) => n.toLocaleString();

function PipelineSection() {
  const [activeStep, setActiveStep] = useState(null);
  return (
    <div style={{ margin: "0 0 80px 0" }}>
      <h2 style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, letterSpacing: 3, color: COPPER, textTransform: "uppercase", marginBottom: 8 }}>The Pipeline</h2>
      <p style={{ color: MUTED, fontSize: 15, marginBottom: 36, maxWidth: 540, lineHeight: 1.6 }}>
        From physics simulation to instant prediction in five steps. The entire pipeline runs on a laptop or a Mac Mini.
      </p>
      <div style={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
        {pipelineSteps.map((step, i) => {
          const isActive = activeStep === i;
          return (
            <div
              key={i}
              onClick={() => setActiveStep(isActive ? null : i)}
              style={{
                flex: "1 1 0",
                minWidth: 130,
                background: isActive ? "#F0EDE8" : CARD_BG,
                borderRadius: 8,
                padding: "20px 16px",
                cursor: "pointer",
                border: `1px solid ${isActive ? COPPER : BORDER}`,
                transition: "all 0.3s ease",
                position: "relative",
              }}
            >
              {i < pipelineSteps.length - 1 && (
                <div style={{ position: "absolute", right: -8, top: "50%", transform: "translateY(-50%)", color: "#CCC", fontSize: 18, zIndex: 1 }}>→</div>
              )}
              <div style={{ fontSize: 28, marginBottom: 8 }}>{step.icon}</div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, fontWeight: 700, color: WARM_WHITE, marginBottom: 4 }}>{step.title}</div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: COPPER }}>{step.time}</div>
              <div style={{
                maxHeight: isActive ? 80 : 0,
                overflow: "hidden",
                transition: "max-height 0.3s ease",
                marginTop: isActive ? 10 : 0,
              }}>
                <div style={{ fontSize: 12, color: MUTED, lineHeight: 1.5 }}>{step.detail}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ScenarioCard({ id, scenario, isSelected, onToggle, baselineElec, baselineGas }) {
  const totalElec = scenario.elec.reduce((a, b) => a + b, 0);
  const totalGas = scenario.gas.reduce((a, b) => a + b, 0);
  const isBaseline = id === "baseline";
  const elecChange = !isBaseline ? ((totalElec - baselineElec) / baselineElec * 100) : 0;
  const gasChange = !isBaseline ? ((totalGas - baselineGas) / baselineGas * 100) : 0;

  return (
    <div
      onClick={onToggle}
      style={{
        background: isSelected ? "#F5F3EF" : CARD_BG,
        borderRadius: 8,
        padding: "16px 18px",
        cursor: "pointer",
        border: `1.5px solid ${isSelected ? scenario.color : BORDER}`,
        transition: "all 0.25s ease",
        opacity: isSelected ? 1 : 0.55,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <div style={{ width: 10, height: 10, borderRadius: "50%", background: scenario.color, flexShrink: 0 }} />
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, fontWeight: 700, color: WARM_WHITE }}>{scenario.label}</div>
      </div>
      <div style={{ fontSize: 12, color: MUTED, marginBottom: 12, lineHeight: 1.4 }}>{scenario.desc}</div>
      <div style={{ display: "flex", gap: 16 }}>
        <div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: MUTED, marginBottom: 2 }}>Electricity</div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 16, fontWeight: 700, color: WARM_WHITE }}>{fmt(totalElec)}</div>
          {!isBaseline && (
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: elecChange < 0 ? GREEN : RED, marginTop: 2 }}>
              {elecChange > 0 ? "+" : ""}{elecChange.toFixed(1)}%
            </div>
          )}
        </div>
        <div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: MUTED, marginBottom: 2 }}>Gas</div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 16, fontWeight: 700, color: WARM_WHITE }}>{fmt(totalGas)}</div>
          {!isBaseline && (
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: gasChange < 0 ? GREEN : RED, marginTop: 2 }}>
              {gasChange > 0 ? "+" : ""}{gasChange.toFixed(1)}%
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function InsightCallout({ icon, title, text }) {
  return (
    <div style={{ background: "#F8F6F3", border: `1px solid ${BORDER}`, borderLeft: `3px solid ${COPPER}`, borderRadius: "0 6px 6px 0", padding: "16px 20px", marginBottom: 12 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
        <span style={{ fontSize: 18, flexShrink: 0, marginTop: 1 }}>{icon}</span>
        <div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, fontWeight: 700, color: WARM_WHITE, marginBottom: 4 }}>{title}</div>
          <div style={{ fontSize: 13, color: MUTED, lineHeight: 1.55 }}>{text}</div>
        </div>
      </div>
    </div>
  );
}

export default function ANESurrogateSite() {
  const [selected, setSelected] = useState(new Set(["baseline", "all"]));
  const [fuelView, setFuelView] = useState("elec");

  const baselineElec = scenarios.baseline.elec.reduce((a, b) => a + b, 0);
  const baselineGas = scenarios.baseline.gas.reduce((a, b) => a + b, 0);

  const toggle = (id) => {
    const next = new Set(selected);
    if (id === "baseline") return;
    if (next.has(id)) { if (next.size > 1) next.delete(id); }
    else next.add(id);
    setSelected(next);
  };

  const chartData = monthNames.map((m, i) => {
    const row = { month: m };
    for (const [id, s] of Object.entries(scenarios)) {
      if (selected.has(id)) {
        row[id] = fuelView === "elec" ? s.elec[i] : Math.max(0, s.gas[i]);
      }
    }
    return row;
  });

  return (
    <div style={{ background: DARK_BG, minHeight: "100vh", color: WARM_WHITE, fontFamily: "'Source Serif 4', Georgia, serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Source+Serif+4:wght@400;600;700&display=swap" rel="stylesheet" />

      {/* Hero */}
      <div style={{ padding: "60px 32px 40px", maxWidth: 900, margin: "0 auto" }}>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: 3, color: COPPER, textTransform: "uppercase", marginBottom: 16 }}>
          Capstone Project — Counterfactual Designs
        </div>
        <h1 style={{ fontSize: 38, fontWeight: 700, lineHeight: 1.2, marginBottom: 16, letterSpacing: -0.5, maxWidth: 700 }}>
          Building Energy Predictions<br />
          <span style={{ color: COPPER }}>at 30,000 per Second</span>
        </h1>
        <p style={{ fontSize: 17, color: MUTED, maxWidth: 600, lineHeight: 1.7, marginBottom: 8 }}>
          A neural network surrogate for EnergyPlus, running on Apple's Neural Engine. 
          It captures the interactive effects between retrofit measures that rules of thumb miss — and responds instantly.
        </p>
        <div style={{ display: "flex", gap: 24, marginTop: 28, flexWrap: "wrap" }}>
          {[
            ["0.033 ms", "per prediction"],
            ["5.1%", "electricity RMSE"],
            ["12 KB", "model size"],
            ["4", "MCP tools"],
          ].map(([val, label], i) => (
            <div key={i} style={{ minWidth: 100 }}>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 26, fontWeight: 700, color: COPPER }}>{val}</div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: MUTED, marginTop: 2 }}>{label}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 32px" }}><div style={{ height: 1, background: BORDER }} /></div>

      {/* Pipeline */}
      <div style={{ padding: "48px 32px", maxWidth: 900, margin: "0 auto" }}>
        <PipelineSection />
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 32px" }}><div style={{ height: 1, background: BORDER }} /></div>

      {/* Interactive Scenario Comparison */}
      <div style={{ padding: "48px 32px", maxWidth: 900, margin: "0 auto" }}>
        <h2 style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, letterSpacing: 3, color: COPPER, textTransform: "uppercase", marginBottom: 8 }}>ECM Comparison</h2>
        <p style={{ color: MUTED, fontSize: 15, marginBottom: 28, maxWidth: 540, lineHeight: 1.6 }}>
          511 m² office building in Chicago. Select scenarios to compare monthly consumption. All predictions generated by the ANE surrogate.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10, marginBottom: 28 }}>
          {Object.entries(scenarios).map(([id, s]) => (
            <ScenarioCard
              key={id} id={id} scenario={s}
              isSelected={selected.has(id)}
              onToggle={() => toggle(id)}
              baselineElec={baselineElec}
              baselineGas={baselineGas}
            />
          ))}
        </div>

        <div style={{ display: "flex", gap: 4, marginBottom: 20 }}>
          {[["elec", "Electricity (kWh)"], ["gas", "Natural Gas (kWh)"]].map(([key, label]) => (
            <button
              key={key}
              onClick={() => setFuelView(key)}
              style={{
                fontFamily: "'JetBrains Mono', monospace", fontSize: 12,
                padding: "8px 16px", borderRadius: 6, border: "none", cursor: "pointer",
                background: fuelView === key ? COPPER : "#EDEBE7",
                color: fuelView === key ? "#FFFFFF" : MUTED,
                fontWeight: fuelView === key ? 700 : 400,
                transition: "all 0.2s ease",
              }}
            >{label}</button>
          ))}
        </div>

        <div style={{ background: CARD_BG, borderRadius: 10, padding: "24px 16px 16px", border: `1px solid ${BORDER}` }}>
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={chartData} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
              <defs>
                {Object.entries(scenarios).map(([id, s]) => (
                  <linearGradient key={id} id={`grad-${id}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={s.color} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={s.color} stopOpacity={0.02} />
                  </linearGradient>
                ))}
              </defs>
              <XAxis dataKey="month" tick={{ fill: MUTED, fontSize: 11, fontFamily: "JetBrains Mono" }} axisLine={{ stroke: BORDER }} tickLine={false} />
              <YAxis tick={{ fill: MUTED, fontSize: 11, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} width={52} tickFormatter={(v) => `${(v/1000).toFixed(1)}k`} />
              <Tooltip
                contentStyle={{ background: "#FFFFFF", border: `1px solid ${BORDER}`, borderRadius: 6, fontFamily: "JetBrains Mono", fontSize: 12, boxShadow: "0 2px 8px rgba(0,0,0,0.1)" }}
                labelStyle={{ color: WARM_WHITE, marginBottom: 4 }}
                itemStyle={{ padding: "1px 0" }}
                formatter={(val, name) => [`${fmt(val)} kWh`, scenarios[name]?.label || name]}
              />
              {Object.entries(scenarios).map(([id, s]) =>
                selected.has(id) ? (
                  <Area
                    key={id} type="monotone" dataKey={id}
                    stroke={s.color} strokeWidth={2}
                    fill={`url(#grad-${id})`}
                    dot={false}
                    animationDuration={600}
                  />
                ) : null
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 32px" }}><div style={{ height: 1, background: BORDER }} /></div>

      {/* Insights */}
      <div style={{ padding: "48px 32px", maxWidth: 900, margin: "0 auto" }}>
        <h2 style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, letterSpacing: 3, color: COPPER, textTransform: "uppercase", marginBottom: 8 }}>What the Surrogate Reveals</h2>
        <p style={{ color: MUTED, fontSize: 15, marginBottom: 28, maxWidth: 540, lineHeight: 1.6 }}>
          Interactive effects between measures that simplified calculations miss.
        </p>

        <InsightCallout
          icon="💡"
          title="LEDs save electricity but increase gas"
          text="Reducing lighting power from 10.76 to 6.0 W/m² cuts electricity 29% but increases gas 44%. Removed lighting waste heat must be replaced by the heating system. This interaction is invisible to measure-by-measure analysis."
        />
        <InsightCallout
          icon="❄️"
          title="HVAC upgrade is a weak standalone ECM"
          text="Improving cooling COP from 3.67 to 5.0 saves only 1.3% on electricity. The cooling load in a 511 m² Chicago office isn't dominant enough to justify the investment alone."
        />
        <InsightCallout
          icon="🏗️"
          title="Envelope work offsets the lighting heat penalty"
          text="Better wall insulation and air sealing cut gas by 22%. When paired with LEDs, they partially compensate for the lost lighting heat — reducing the net gas increase from +44% to +18%."
        />
        <InsightCallout
          icon="🎯"
          title="The counterfactual matters"
          text="What would this building have consumed without the retrofit? That question — the counterfactual — is the foundation of every savings claim. A competent counterfactual requires understanding uncertainty, not just passing a threshold test."
        />
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 32px" }}><div style={{ height: 1, background: BORDER }} /></div>

      {/* MCP Tools */}
      <div style={{ padding: "48px 32px 28px", maxWidth: 900, margin: "0 auto" }}>
        <h2 style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, letterSpacing: 3, color: COPPER, textTransform: "uppercase", marginBottom: 8 }}>MCP Tools</h2>
        <p style={{ color: MUTED, fontSize: 15, marginBottom: 28, maxWidth: 540, lineHeight: 1.6 }}>
          The surrogate exposes four tools via the Model Context Protocol. Any AI assistant can call them directly — no file management, no simulation queue.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 10 }}>
          {[
            { name: "predict_energy", desc: "Single month, single parameter set → electricity and gas" },
            { name: "compare_scenarios", desc: "2–4 ECM packages across all 12 months" },
            { name: "sweep_parameter", desc: "Sensitivity analysis — one parameter across its full range" },
            { name: "get_parameter_info", desc: "Valid ranges and baseline values for all inputs" },
          ].map((tool, i) => (
            <div key={i} style={{ background: CARD_BG, border: "1px solid #2A2A2A", borderRadius: 8, padding: "18px 16px" }}>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, fontWeight: 700, color: COPPER, marginBottom: 8 }}>{tool.name}</div>
              <div style={{ fontSize: 13, color: MUTED, lineHeight: 1.5 }}>{tool.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Tech Specs */}
      <div style={{ padding: "28px 32px 48px", maxWidth: 900, margin: "0 auto" }}>
        <div style={{ background: CARD_BG, border: "1px solid #2A2A2A", borderRadius: 10, padding: "24px 28px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 20 }}>
          {[
            ["Building", "DOE Reference Small Office, 511 m², 5-zone"],
            ["Location", "Chicago, IL (TMY3 weather)"],
            ["Training Data", "50 EnergyPlus 25.2 runs, 600 monthly rows"],
            ["Architecture", "MLP: 5 → 64 → 64 → 32 → 2"],
            ["Accuracy", "Electricity RMSE: 5.1% — Gas RMSE: 32.1%"],
            ["Inference", "0.033 ms/prediction on Apple M4 Neural Engine"],
          ].map(([label, value], i) => (
            <div key={i}>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: COPPER, marginBottom: 4 }}>{label}</div>
              <div style={{ fontSize: 13, color: MUTED, lineHeight: 1.5 }}>{value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div style={{ padding: "32px 32px 48px", maxWidth: 900, margin: "0 auto", textAlign: "center" }}>
        <div style={{ height: 1, background: BORDER, marginBottom: 32 }} />
        <div style={{ display: "flex", justifyContent: "center", gap: 24, flexWrap: "wrap", marginBottom: 16 }}>
          <a href="https://cfdesigns.vercel.app" target="_blank" rel="noopener noreferrer"
            style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#b5632e", textDecoration: "none", fontWeight: 600 }}>
            Counterfactual Designs →
          </a>
          <a href="https://bayesian-mv.vercel.app" target="_blank" rel="noopener noreferrer"
            style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#7c5cbf", textDecoration: "none", fontWeight: 600 }}>
            Bayesian Module →
          </a>
          <a href="https://mv-course.vercel.app" target="_blank" rel="noopener noreferrer"
            style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#a67c28", textDecoration: "none", fontWeight: 600 }}>
            IPMVP Reference →
          </a>
          <a href="https://mv-classmap.vercel.app" target="_blank" rel="noopener noreferrer"
            style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#2d7d46", textDecoration: "none", fontWeight: 600 }}>
            Learning Path →
          </a>
          <a href="https://cmvp-capstone.vercel.app" target="_blank" rel="noopener noreferrer"
            style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#c0392b", textDecoration: "none", fontWeight: 600 }}>
            CMVP Capstone →
          </a>
          <a href="https://mnvscore.vercel.app" target="_blank" rel="noopener noreferrer"
            style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#2c6fad", textDecoration: "none", fontWeight: 600 }}>
            M&V Scorecard →
          </a>
        </div>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: MUTED, marginBottom: 8 }}>
          <a href="https://counterfactual-designs.com" target="_blank" rel="noopener noreferrer"
            style={{ color: MUTED, textDecoration: "none" }}>
            counterfactual-designs.com
          </a>
        </div>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#AAA" }}>
          github.com/jskromer/ane-surrogate
        </div>
      </div>
    </div>
  );
}
