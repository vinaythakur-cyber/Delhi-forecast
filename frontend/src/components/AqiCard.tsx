"use client";

import type { Current } from "@/lib/api";
import { fixed, istFull } from "@/lib/format";

// Colours that need dark text for contrast (light yellow / light green).
const DARK_TEXT = new Set(["Satisfactory", "Moderate"]);

function barColor(value: number | null, cats: Current["categories"]): string {
  if (value === null) return "var(--muted)";
  return (cats.find((c) => value <= c.high) ?? cats[cats.length - 1]).color;
}

export function AqiCard({ data }: { data: Current }) {
  const { aqi, pollutants, weather, categories } = data;
  const change = aqi.change_3h;
  return (
    <section className="card" aria-labelledby="aqi-title">
      <div className="card-head">
        <h2 id="aqi-title">Air quality now</h2>
        <span className="sub">{data.location.name}</span>
      </div>

      <div className="aqi-hero">
        <div className="aqi-num" style={{ color: aqi.color }} aria-label={`AQI ${aqi.value}`}>{aqi.value}</div>
        <div>
          <span className={`aqi-cat ${DARK_TEXT.has(aqi.category) ? "dark-text" : ""}`} style={{ background: aqi.color }}>
            {aqi.category}
          </span>
          <div className="note" style={{ marginTop: 6 }}>
            {change === null ? "" : change === 0 ? "Unchanged over 3 h" : `${change > 0 ? "▲" : "▼"} ${Math.abs(change)} vs 3 h ago`}
          </div>
        </div>
      </div>

      <p style={{ marginTop: 14 }}>{aqi.advice}</p>

      <div className="subidx" aria-label="Sub-index of each pollutant">
        {pollutants.map((p) => (
          <div key={p.key} className={`subidx-row ${p.counts_toward_aqi ? "" : "off"}`} title={p.counts_toward_aqi ? "Counts toward the AQI" : "Shown for context; does not set the AQI"}>
            <span>{p.label}</span>
            <div className="bar"><i style={{ width: `${Math.min(100, ((p.sub_index ?? 0) / 500) * 100)}%`, background: barColor(p.sub_index, categories) }} /></div>
            <span style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{p.sub_index ?? "–"}</span>
          </div>
        ))}
      </div>
      <p className="note" style={{ marginTop: 10 }}>{aqi.basis_note} Faded rows do not count toward the number above.</p>

      <dl className="kv" style={{ marginTop: 12 }}>
        <dt>PM2.5 now</dt><dd>{fixed(pollutants[0].value)} µg/m³</dd>
        <dt>PM2.5 24 h average</dt><dd>{fixed(pollutants[0].average)} µg/m³</dd>
        {weather.lag_hours !== null && weather.lag_hours > 6 && (
          <>
            <dt style={{ gridColumn: "1 / -1", marginTop: 6 }} className="note">Weather feed is behind: values are as of {istFull(weather.as_of)}</dt>
          </>
        )}
        <dt>Temperature</dt><dd>{fixed(weather.temperature_c)} °C</dd>
        <dt>Humidity</dt><dd>{fixed(weather.humidity_pct, 0)} %</dd>
        <dt>Wind</dt><dd>{fixed(weather.wind_kmh)} km/h</dd>
        <dt>Mixing-layer height</dt><dd>{fixed(weather.boundary_layer_m, 0)} m</dd>
      </dl>
    </section>
  );
}
