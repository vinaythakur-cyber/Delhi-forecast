"use client";

import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Header } from "@/components/Header";
import { api, type HorizonTable, type ModelMetrics } from "@/lib/api";
import { fixed, ist, pct } from "@/lib/format";
import { useApi } from "@/lib/useApi";

const COLORS = { persistence: "#8a8780", seasonal_naive: "#eb6834", lightgbm: "#2a78d6", sarima: "#1baf7a" } as const;
const NAMES: Record<string, string> = { persistence: "Persistence", seasonal_naive: "Seasonal naive (24 h)", lightgbm: "LightGBM (this project)", sarima: "SARIMA" };
type Target = "pm2_5" | "pm10";
type Measure = "mae" | "rmse";

function horizonRows(table: HorizonTable, measure: Measure) {
  const hs = Object.keys(table.lightgbm ?? table.persistence).map(Number).sort((a, b) => a - b);
  return hs.map((h) => {
    const row: Record<string, number> = { h };
    for (const m of Object.keys(table)) row[m] = table[m][String(h)]?.[measure];
    return row;
  });
}

function Seg<T extends string>({ value, onChange, options, label }: { value: T; onChange: (v: T) => void; options: { key: T; label: string }[]; label: string }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.key} aria-pressed={value === o.key} onClick={() => onChange(o.key)}>{o.label}</button>
      ))}
    </div>
  );
}

export default function ModelPage() {
  const { data, error, loading, status } = useApi<ModelMetrics>(() => api<ModelMetrics>("/model-metrics"), []);
  const [target, setTarget] = useState<Target>("pm2_5");
  const [measure, setMeasure] = useState<Measure>("mae");

  return (
    <>
      <Header active="model" />
      <main className="shell">
        <div style={{ marginTop: 22 }}>
          <h1 style={{ fontSize: "1.8rem" }}>How good is the forecast?</h1>
          <p className="note" style={{ maxWidth: "70ch", marginTop: 8 }}>
            Every number here comes from <b>walk-forward validation</b>: the model is trained only on the past, forecasts the following weeks, then
            the cut-off moves forward and it repeats. There is no random train/test split. Each model is scored on exactly the same hours and compared
            with two models that need no training: &quot;nothing changes&quot; (persistence) and &quot;same hour yesterday&quot; (seasonal naive).
          </p>
        </div>
        {error && <div className="err" style={{ marginTop: 16 }} role="alert">{status === 503 ? "No evaluation is available yet. Run `python -m ml.evaluate`." : `Could not load metrics: ${error}`}</div>}
        {loading && !data && <div className="card" style={{ marginTop: 16 }}><div className="skeleton" /></div>}
        {data && <Content data={data} target={target} setTarget={setTarget} measure={measure} setMeasure={setMeasure} />}
      </main>
    </>
  );
}

function Content({ data, target, setTarget, measure, setMeasure }: { data: ModelMetrics; target: Target; setTarget: (t: Target) => void; measure: Measure; setMeasure: (m: Measure) => void }) {
  const ev = data.evaluation;
  const t = ev.targets[target];
  const rows = horizonRows(t.by_horizon, measure);
  const horizons = rows.map((r) => r.h);
  const best = (h: number) => Object.keys(t.by_horizon).reduce((a, m) => (t.by_horizon[m][String(h)].mae < t.by_horizon[a][String(h)].mae ? m : a));
  const g = (h: number, m: string) => t.by_horizon[m]?.[String(h)]?.mae;
  const skill24 = t.skill_vs_persistence.lightgbm["24"];
  const skill72 = t.skill_vs_persistence.lightgbm["72"];
  const skill1 = t.skill_vs_persistence.lightgbm["1"];
  const cov24 = t.interval.coverage["24"];
  const skillRows = horizons.map((h) => ({ h, lightgbm: t.skill_vs_persistence.lightgbm[String(h)], seasonal_naive: t.skill_vs_persistence.seasonal_naive[String(h)] }));
  const covRows = horizons.map((h) => ({ h, all: t.interval.coverage[String(h)], winter: t.interval_winter.coverage[String(h)] }));
  const folds = t.by_fold.map((f) => ({ ...f, label: ist(String(f.cutoff), { month: "short", year: "2-digit" }) }));
  const maxImp = Math.max(...ev.feature_importance.map((f) => f[1]), 0.001);

  return (
    <div className="grid">
      <section className="card">
        <div className="card-head">
          <h2>Headline results</h2>
          <span className="sub">{ev.scope}</span>
          <div className="right">
            <Seg value={target} onChange={setTarget} label="Pollutant" options={[{ key: "pm2_5", label: "PM2.5" }, { key: "pm10", label: "PM10" }]} />
          </div>
        </div>
        <div className="stat-row">
          <div className={`stat ${skill24 > 0 ? "good" : "bad"}`}><b>{pct(skill24, 0)}</b><span>lower error than persistence at 24 h ({fixed(g(24, "lightgbm"))} vs {fixed(g(24, "persistence"))} µg/m³)</span></div>
          <div className={`stat ${skill72 > 0 ? "good" : "bad"}`}><b>{pct(skill72, 0)}</b><span>lower error than persistence at 72 h ({fixed(g(72, "lightgbm"))} vs {fixed(g(72, "persistence"))} µg/m³)</span></div>
          <div className={`stat ${skill1 < -0.02 ? "bad" : skill1 > 0.02 ? "good" : ""}`}><b>{pct(skill1, 0)}</b><span>at 1 h: {skill1 < -0.02 ? "persistence is better. Air barely changes in an hour, so there is little to learn" : skill1 > 0.02 ? "a small edge over persistence" : "a tie with persistence. Air barely changes in an hour, so there is little to learn"}</span></div>
          <div className="stat"><b>{pct(cov24, 0)}</b><span>of actual values fell inside the 80% interval at 24 h (target 80%)</span></div>
        </div>
        <p className="note" style={{ marginTop: 12 }}>
          Evaluated on {ev.folds.length} walk-forward folds ({ev.config.test_days}-day test windows, forecast origins every {ev.config.origin_step_h} h), data from{" "}
          {ist(ev.data_range.start, { day: "numeric", month: "short", year: "numeric" })} to {ist(ev.data_range.end, { day: "numeric", month: "short", year: "numeric" })}.
          Generated {ist(ev.generated_at, { day: "numeric", month: "short", year: "numeric" })}{ev.source === "shipped" ? " (shipped with the project; it is recomputed when you run `python -m ml.evaluate`)" : ""}.
        </p>
      </section>

      <section className="card">
        <div className="card-head">
          <h2>Error by forecast horizon</h2>
          <span className="sub">Lower is better. Hours ahead on the x-axis.</span>
          <div className="right"><Seg value={measure} onChange={setMeasure} label="Error measure" options={[{ key: "mae", label: "MAE" }, { key: "rmse", label: "RMSE" }]} /></div>
        </div>
        <div style={{ width: "100%", height: 320 }}>
          <ResponsiveContainer>
            <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="h" type="number" ticks={horizons} domain={[0, 72]} stroke="var(--muted)" tick={{ fontSize: 12 }} label={{ value: "hours ahead", position: "insideBottom", offset: -10, fill: "var(--ink-2)", fontSize: 12 }} />
              <YAxis stroke="var(--muted)" width={44} tick={{ fontSize: 12 }} label={{ value: "µg/m³", angle: -90, position: "insideLeft", fill: "var(--ink-2)", fontSize: 12 }} />
              <Tooltip formatter={(v) => `${fixed(Number(v))} µg/m³`} labelFormatter={(h) => `${h} h ahead`} />
              <Legend verticalAlign="top" height={32} formatter={(v) => NAMES[String(v)] ?? String(v)} />
              {(["persistence", "seasonal_naive", "lightgbm"] as const).map((m) => (
                <Line key={m} dataKey={m} stroke={COLORS[m]} strokeWidth={m === "lightgbm" ? 3 : 2} dot={{ r: 3 }} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="scroll-x" style={{ marginTop: 10 }}>
          <table className="data">
            <caption className="note" style={{ textAlign: "left", paddingBottom: 6 }}>Mean absolute error (µg/m³). Bold green marks the best model at each horizon.</caption>
            <thead><tr><th>Horizon</th>{Object.keys(t.by_horizon).map((m) => <th key={m}>{NAMES[m]}</th>)}<th>Hours scored</th></tr></thead>
            <tbody>
              {horizons.map((h) => (
                <tr key={h}>
                  <td>{h} h</td>
                  {Object.keys(t.by_horizon).map((m) => <td key={m} className={best(h) === m ? "best" : ""}>{fixed(g(h, m))}</td>)}
                  <td>{t.by_horizon.lightgbm[String(h)].n.toLocaleString("en-IN")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <div className="card-head"><h2>Skill versus persistence</h2><span className="sub">Positive = lower error than &quot;nothing changes&quot;. Negative = worse.</span></div>
        <div style={{ width: "100%", height: 260 }}>
          <ResponsiveContainer>
            <BarChart data={skillRows} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="h" stroke="var(--muted)" tick={{ fontSize: 12 }} tickFormatter={(h) => `${h} h`} />
              <YAxis stroke="var(--muted)" width={48} tick={{ fontSize: 12 }} tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`} />
              <ReferenceLine y={0} stroke="var(--ink-2)" />
              <Tooltip formatter={(v) => pct(Number(v), 1)} labelFormatter={(h) => `${h} h ahead`} />
              <Legend verticalAlign="top" height={32} formatter={(v) => NAMES[String(v)] ?? String(v)} />
              <Bar dataKey="seasonal_naive" fill={COLORS.seasonal_naive} radius={[3, 3, 0, 0]} maxBarSize={22} isAnimationActive={false} />
              <Bar dataKey="lightgbm" fill={COLORS.lightgbm} radius={[3, 3, 0, 0]} maxBarSize={22} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="card">
        <div className="card-head"><h2>Are the uncertainty bands honest?</h2><span className="sub">The 80% interval should contain the truth about 80% of the time.</span></div>
        <div style={{ width: "100%", height: 260 }}>
          <ResponsiveContainer>
            <LineChart data={covRows} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="h" stroke="var(--muted)" tick={{ fontSize: 12 }} tickFormatter={(h) => `${h} h`} />
              <YAxis domain={[0.4, 1]} stroke="var(--muted)" width={48} tick={{ fontSize: 12 }} tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`} />
              <ReferenceLine y={0.8} stroke="var(--ink-2)" strokeDasharray="5 4" label={{ value: "target 80%", position: "insideTopRight", fill: "var(--ink-2)", fontSize: 12 }} />
              <Tooltip formatter={(v) => pct(Number(v), 0)} labelFormatter={(h) => `${h} h ahead`} />
              <Legend verticalAlign="top" height={32} formatter={(v) => (v === "all" ? "All seasons" : "Winter only (Nov to Feb)")} />
              <Line dataKey="all" stroke={COLORS.lightgbm} strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />
              <Line dataKey="winter" stroke={COLORS.seasonal_naive} strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="note">
          Intervals are quantile forecasts widened by split-conformal calibration on held-out data. Coverage below the dashed line means the band is too narrow (over-confident); above it means too wide.
          Mean width at 24 h: {fixed(t.interval.mean_width["24"])} µg/m³.
        </p>
      </section>

      <section className="card">
        <div className="card-head"><h2>Winter versus the rest of the year</h2><span className="sub">MAE (µg/m³). Winter air is both dirtier and harder to forecast.</span></div>
        <div className="scroll-x">
          <table className="data">
            <thead><tr><th>Horizon</th><th>Winter: persistence</th><th>Winter: LightGBM</th><th>Other: persistence</th><th>Other: LightGBM</th></tr></thead>
            <tbody>
              {horizons.map((h) => {
                const w = t.by_season.winter.by_horizon, o = t.by_season.other.by_horizon;
                return (
                  <tr key={h}>
                    <td>{h} h</td>
                    <td>{fixed(w.persistence?.[String(h)]?.mae)}</td><td>{fixed(w.lightgbm?.[String(h)]?.mae)}</td>
                    <td>{fixed(o.persistence?.[String(h)]?.mae)}</td><td>{fixed(o.lightgbm?.[String(h)]?.mae)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="note" style={{ marginTop: 8 }}>Winter windows scored: {t.by_season.winter.n.toLocaleString("en-IN")} forecasts · other seasons: {t.by_season.other.n.toLocaleString("en-IN")}.</p>
      </section>

      <section className="card">
        <div className="card-head"><h2>Does it hold across time?</h2><span className="sub">Average MAE over all horizons in each walk-forward fold, labelled by the cut-off month.</span></div>
        <div style={{ width: "100%", height: 280 }}>
          <ResponsiveContainer>
            <ComposedChart data={folds} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="label" stroke="var(--muted)" tick={{ fontSize: 12 }} />
              <YAxis stroke="var(--muted)" width={44} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v) => `${fixed(Number(v))} µg/m³`} />
              <Legend verticalAlign="top" height={32} formatter={(v) => NAMES[String(v)] ?? String(v)} />
              {(["persistence", "seasonal_naive", "lightgbm"] as const).map((m) => (
                <Bar key={m} dataKey={m} fill={COLORS[m]} radius={[3, 3, 0, 0]} maxBarSize={14} isAnimationActive={false} />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </section>

      {ev.sarima && (
        <section className="card">
          <div className="card-head"><h2>Classical baseline: SARIMA</h2><span className="sub">{ev.sarima.scope}</span></div>
          <div className="scroll-x">
            <table className="data">
              <thead><tr><th>Horizon</th>{Object.keys(ev.sarima.by_horizon).map((m) => <th key={m}>{NAMES[m] ?? m}</th>)}</tr></thead>
              <tbody>
                {Object.keys(ev.sarima.by_horizon.lightgbm).sort((a, b) => Number(a) - Number(b)).map((h) => {
                  const mins = Math.min(...Object.keys(ev.sarima!.by_horizon).map((m) => ev.sarima!.by_horizon[m][h].mae));
                  return (
                    <tr key={h}>
                      <td>{h} h</td>
                      {Object.keys(ev.sarima!.by_horizon).map((m) => <td key={m} className={ev.sarima!.by_horizon[m][h].mae === mins ? "best" : ""}>{fixed(ev.sarima!.by_horizon[m][h].mae)}</td>)}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="note" style={{ marginTop: 8 }}>SARIMA is slow, so it runs on a subset of origins (one per day in every second fold); all four models are scored on those same {ev.sarima.n_rows.toLocaleString("en-IN")} forecasts.</p>
        </section>
      )}

      <section className="card">
        <div className="card-head"><h2>What the model looks at</h2><span className="sub">Share of the PM2.5 median model&apos;s total gain (importance), top features.</span></div>
        <div style={{ display: "grid", gap: 6 }}>
          {ev.feature_importance.map(([name, v]) => (
            <div key={name} style={{ display: "grid", gridTemplateColumns: "minmax(120px, 190px) 1fr 48px", gap: 10, alignItems: "center", fontSize: "0.88rem" }}>
              <code style={{ fontFamily: "var(--mono)" }}>{name}</code>
              <div className="bar"><i style={{ width: `${(v / maxImp) * 100}%`, background: "var(--accent)" }} /></div>
              <span style={{ textAlign: "right" }}>{pct(v, 1)}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <div className="card-head"><h2>Method and limitations</h2></div>
        <h3>Method</h3>
        <ul className="limits">{ev.procedure.map((p) => <li key={p}>{p}</li>)}</ul>
        <h3 style={{ marginTop: 16 }}>Known limitations</h3>
        <ul className="limits">{data.limitations.map((p) => <li key={p}>{p}</li>)}</ul>
        {data.production_model && <p className="note" style={{ marginTop: 14 }}>Production model last trained {ist(data.production_model.trained_at, { day: "numeric", month: "short", year: "numeric", hour: "numeric", minute: "2-digit", hour12: true })} IST. Headline AQI basis: <code>{data.aqi_basis}</code>.</p>}
      </section>
    </div>
  );
}
