"use client";

import { useState } from "react";
import { api, type Category, type Seasonal } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { fixed } from "@/lib/format";

type Metric = "aqi" | "pm2_5";

// Sequential scale (pure presentation): pale -> deep red.
function sequential(v: number, lo: number, hi: number): string {
  const t = Math.max(0, Math.min(1, (v - lo) / Math.max(1e-9, hi - lo)));
  const a = [254, 240, 217];
  const b = [153, 0, 13];
  const c = a.map((x, i) => Math.round(x + (b[i] - x) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function range(values: (number | null)[]): [number, number] {
  const v = values.filter((x): x is number => x !== null);
  return v.length ? [Math.min(...v), Math.max(...v)] : [0, 1];
}

export function SeasonalHeatmap({ location, categories }: { location: string; categories: Category[] }) {
  const [metric, setMetric] = useState<Metric>("aqi");
  // The AQI is a trailing 24-hour average, so it has no hour-of-day pattern. The hourly grid therefore
  // always shows PM2.5 concentration; the toggle only changes the year-by-month grid.
  const hourly = useApi<Seasonal>(() => api<Seasonal>("/seasonal", { location, metric: "pm2_5" }), [location]);
  const monthly = useApi<Seasonal>(() => api<Seasonal>("/seasonal", { location, metric }), [location, metric]);

  const err = hourly.error || monthly.error;
  const h = hourly.data;
  const m = monthly.data;
  const [hLo, hHi] = h ? range(h.month_by_hour.flat()) : [0, 1];
  const years = m ? Array.from(new Set(m.year_month.map((x) => x.year))).sort() : [];
  const ym = new Map((m?.year_month ?? []).map((x) => [`${x.year}-${x.month}`, x.value]));
  const [mLo, mHi] = m ? range(m.year_month.map((x) => x.value)) : [0, 1];
  const monthColor = (v: number | null) => {
    if (v === null) return "var(--surface-2)";
    return metric === "aqi" ? (categories.find((c) => v <= c.high) ?? categories[categories.length - 1] ?? { color: "#888" }).color : sequential(v, mLo, mHi);
  };

  return (
    <section className="card" aria-labelledby="seas-title">
      <div className="card-head">
        <h2 id="seas-title">Seasonal patterns</h2>
        <span className="sub">When is the air worst? Averages over all available history</span>
      </div>
      {err && <div className="err">Could not load seasonal data: {err}</div>}
      {!err && (!h || !m ? <div className="skeleton" /> : (
        <div style={{ display: "grid", gap: 24 }}>
          <div>
            <h3 style={{ marginBottom: 2 }}>PM2.5 by month and hour of day (IST)</h3>
            <p className="note" style={{ marginBottom: 8 }}>
              Hourly concentration, µg/m³. {fixed(hLo, 0)} (pale) to {fixed(hHi, 0)} (dark red). Late-evening hours and winter months are the worst.
            </p>
            <div className="heat" style={{ gridTemplateColumns: "34px repeat(24, minmax(0, 1fr))" }} role="img" aria-label="Heatmap of average PM2.5 by month and hour of day">
              <span />
              {Array.from({ length: 24 }, (_, i) => <span key={i} className="heat-top">{i % 3 === 0 ? i : ""}</span>)}
              {h.month_by_hour.map((row, mi) => (
                <HeatRow key={mi} label={h.months[mi]} values={row} fill={(v) => (v === null ? "var(--surface-2)" : sequential(v, hLo, hHi))} unit={h.unit} title={(i) => `${h.months[mi]}, ${String(i).padStart(2, "0")}:00`} />
              ))}
            </div>
          </div>

          <div>
            <div className="card-head" style={{ marginBottom: 8 }}>
              <h3>{metric === "aqi" ? "AQI" : "PM2.5"} by year and month</h3>
              <div className="right seg" role="group" aria-label="Year by month metric">
                <button aria-pressed={metric === "aqi"} onClick={() => setMetric("aqi")}>AQI</button>
                <button aria-pressed={metric === "pm2_5"} onClick={() => setMetric("pm2_5")}>PM2.5</button>
              </div>
            </div>
            <div className="heat" style={{ gridTemplateColumns: "34px repeat(12, minmax(0, 1fr))" }} role="img" aria-label="Heatmap of the monthly average by year">
              <span />
              {m.months.map((mo) => <span key={mo} className="heat-top">{mo}</span>)}
              {years.map((y) => (
                <HeatRow key={y} label={String(y)} values={m.months.map((_, i) => ym.get(`${y}-${i + 1}`) ?? null)} fill={monthColor} unit={m.unit} title={(i) => `${m.months[i]} ${y}`} square={false} />
              ))}
            </div>
            <div className="legend" style={{ marginTop: 10, marginBottom: 0 }}>
              {metric === "aqi"
                ? categories.map((c) => <span key={c.name}><i className="box" style={{ background: c.color }} />{c.name}</span>)
                : <span>PM2.5 µg/m³: {fixed(mLo, 0)} (pale) to {fixed(mHi, 0)} (dark red)</span>}
            </div>
          </div>
          <p className="note" style={{ margin: 0 }}>Why evenings? After sunset the mixing layer collapses, so heating, traffic and (in autumn) crop-residue smoke are trapped in a thin layer of cold air. Hover a cell for its value. Grey cells mean no data (the history starts in August 2022).</p>
        </div>
      ))}
    </section>
  );
}

function HeatRow({ label, values, fill, unit, title, square = true }: {
  label: string; values: (number | null)[]; fill: (v: number | null) => string; unit: string; title: (i: number) => string; square?: boolean;
}) {
  return (
    <>
      <span className="heat-label">{label}</span>
      {values.map((v, i) => (
        <span key={i} className="heat-cell" style={{ background: fill(v), aspectRatio: square ? "1 / 1" : "2 / 1" }} title={`${title(i)}: ${v === null ? "no data" : `${fixed(v, 0)} ${unit}`}`} />
      ))}
    </>
  );
}
