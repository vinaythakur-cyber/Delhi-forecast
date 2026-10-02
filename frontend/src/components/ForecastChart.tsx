"use client";

import { useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Forecast } from "@/lib/api";
import { fixed, istDay, istFull, istHour } from "@/lib/format";

type Mode = "aqi" | "pm2_5";

interface Row {
  t: number;
  obs?: number | null;
  q50?: number | null;
  band?: [number, number] | null;
  category?: string | null;
  horizon?: number;
}

function Tip({ active, payload, unit }: { active?: boolean; payload?: ReadonlyArray<{ payload: Row }>; unit: string }) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload;
  const when = new Date(r.t).toISOString();
  return (
    <div className="tip">
      <div>{istFull(when)}</div>
      {r.obs !== undefined && r.obs !== null && r.horizon === undefined && <div>Observed: <b>{fixed(r.obs, 0)}</b> {unit}</div>}
      {r.horizon !== undefined && r.q50 !== undefined && r.q50 !== null && (
        <>
          <div>+{r.horizon} h forecast: <b>{fixed(r.q50, 0)}</b> {unit}</div>
          {r.band && <div>80% range: {fixed(r.band[0], 0)} to {fixed(r.band[1], 0)}</div>}
          {r.category && <div>{r.category}</div>}
        </>
      )}
    </div>
  );
}

const IST_OFFSET_MS = 5.5 * 3600 * 1000;
const DAY_MS = 24 * 3600 * 1000;

/** One tick at every midnight IST inside [min, max]. */
function midnightTicks(min: number, max: number): number[] {
  const first = Math.ceil((min + IST_OFFSET_MS) / DAY_MS) * DAY_MS - IST_OFFSET_MS;
  const ticks: number[] = [];
  for (let t = first; t <= max; t += DAY_MS) ticks.push(t);
  return ticks;
}

function valueTicks(mode: Mode, yMax: number): number[] {
  const base = mode === "aqi" ? [0, 50, 100, 200, 300, 400, 500] : null;
  if (base) return base.filter((v) => v <= yMax);
  const step = yMax <= 100 ? 25 : yMax <= 300 ? 50 : 100;
  return Array.from({ length: Math.floor(yMax / step) + 1 }, (_, i) => i * step);
}

export function ForecastChart({ data }: { data: Forecast }) {
  const [mode, setMode] = useState<Mode>("aqi");
  const unit = mode === "aqi" ? "AQI" : "µg/m³";

  const { rows, yMax, peak, issuedMs, xTicks } = useMemo(() => {
    const issuedMs = new Date(data.issued_at).getTime();
    const rows: Row[] = data.recent.map((r) => ({ t: new Date(r.t).getTime(), obs: mode === "aqi" ? r.aqi : r.pm2_5 }));
    const last = rows[rows.length - 1];
    // The forecast fan starts at the last observed value, so the two pieces join up.
    rows[rows.length - 1] = { ...last, q50: last?.obs ?? null, band: last?.obs != null ? [last.obs, last.obs] : null };
    let peak: { v: number; t: string; cat: string | null } | null = null;
    for (const p of data.points) {
      const q = mode === "aqi" ? p.aqi : p.pm2_5;
      if (q.q50 === null || q.q10 === null || q.q90 === null) continue;
      rows.push({ t: new Date(p.target_ts).getTime(), q50: q.q50, band: [q.q10, q.q90], category: p.category, horizon: p.horizon });
      if (!peak || q.q50 > peak.v) peak = { v: q.q50, t: p.target_ts, cat: p.category };
    }
    const top = Math.max(...rows.map((r) => Math.max(r.obs ?? 0, r.band?.[1] ?? 0)));
    return { rows, yMax: Math.ceil((top * 1.08) / 50) * 50, peak, issuedMs, xTicks: midnightTicks(rows[0].t, rows[rows.length - 1].t) };
  }, [data, mode]);

  const bands = mode === "aqi" ? data.categories : data.pm25_category_bands;

  return (
    <section className="card" aria-labelledby="fc-title">
      <div className="card-head">
        <h2 id="fc-title">Next 72 hours</h2>
        <span className="sub">Forecast issued from data through {istFull(data.issued_at)}. Ticks mark midnight IST.</span>
        <div className="right seg" role="group" aria-label="Forecast metric">
          <button aria-pressed={mode === "aqi"} onClick={() => setMode("aqi")}>AQI</button>
          <button aria-pressed={mode === "pm2_5"} onClick={() => setMode("pm2_5")}>PM2.5</button>
        </div>
      </div>

      <div className="legend">
        <span><i style={{ background: "var(--obs)" }} />Observed (last 48 h)</span>
        <span><i style={{ background: "var(--accent)" }} />Forecast (median)</span>
        <span><i className="box" style={{ background: "var(--accent-soft)" }} />80% prediction interval</span>
        {bands.map((b) => (
          <span key={b.name}><i className="box" style={{ background: b.color, opacity: 0.35 }} />{b.name}</span>
        ))}
      </div>

      <div style={{ width: "100%", height: 340 }}>
        <ResponsiveContainer>
          <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            {bands.map((b) => (
              <ReferenceArea key={b.name} y1={b.low} y2={Math.min(b.high, yMax)} fill={b.color} fillOpacity={0.1} ifOverflow="hidden" />
            ))}
            <XAxis
              dataKey="t"
              type="number"
              scale="time"
              domain={["dataMin", "dataMax"]}
              ticks={xTicks}
              tickFormatter={(ms: number) => istDay(new Date(ms).toISOString())}
              stroke="var(--muted)"
              tick={{ fontSize: 12 }}
            />
            <YAxis domain={[0, yMax]} ticks={valueTicks(mode, yMax)} width={44} stroke="var(--muted)" tick={{ fontSize: 12 }} />
            <Tooltip content={<Tip unit={unit} />} />
            <ReferenceLine x={issuedMs} stroke="var(--ink-2)" strokeDasharray="4 3" label={{ value: "now", position: "insideTopLeft", fill: "var(--ink-2)", fontSize: 12 }} />
            <Area dataKey="band" stroke="none" fill="var(--accent)" fillOpacity={0.2} isAnimationActive={false} connectNulls />
            <Line dataKey="obs" stroke="var(--obs)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
            <Line dataKey="q50" stroke="var(--accent)" strokeWidth={2.4} dot={false} isAnimationActive={false} connectNulls />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {peak && (
        <p style={{ marginTop: 10 }}>
          Highest expected value: <b>{fixed(peak.v, 0)} {unit}</b>{peak.cat && mode === "aqi" ? ` (${peak.cat})` : ""} around {istDay(peak.t)}, {istHour(peak.t).split(", ").pop()} IST.
        </p>
      )}
      <p className="note">
        {data.caveat}{" "}
        {data.error_reference && (
          <>
            Typical PM2.5 error on past tests: {Object.entries(data.error_reference).map(([h, e]) => `${e.model_mae} µg/m³ at ${h} h`).join(", ")} (always-the-same-as-now would be{" "}
            {Object.entries(data.error_reference).map(([h, e]) => `${e.persistence_mae} at ${h} h`).join(", ")}).{" "}
          </>
        )}
        <a href="/model/">How accurate is this?</a>
      </p>
    </section>
  );
}
