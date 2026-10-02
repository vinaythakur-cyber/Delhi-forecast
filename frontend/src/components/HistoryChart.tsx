"use client";

import { useMemo, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Category, type History } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { fixed, ist } from "@/lib/format";

const RANGES = [
  { key: "7d", label: "7 days", days: 7, resolution: "hourly" },
  { key: "30d", label: "30 days", days: 30, resolution: "daily" },
  { key: "90d", label: "90 days", days: 90, resolution: "daily" },
  { key: "1y", label: "1 year", days: 365, resolution: "daily" },
  { key: "all", label: "All", days: 3650, resolution: "monthly" },
] as const;

const METRICS = [
  { key: "aqi", label: "AQI" },
  { key: "pm2_5", label: "PM2.5" },
  { key: "pm10", label: "PM10" },
] as const;

interface Row { label: string; ts: number; v: number | null; range: [number, number] | null }

export function HistoryChart({ location, anchor, categories }: { location: string; anchor: string | null; categories: Category[] }) {
  const [range, setRange] = useState<(typeof RANGES)[number]["key"]>("90d");
  const [metric, setMetric] = useState<(typeof METRICS)[number]["key"]>("aqi");
  const r = RANGES.find((x) => x.key === range)!;

  const { data, error, loading } = useApi<History>(
    () => {
      const end = anchor ? new Date(anchor) : new Date();
      const start = new Date(end.getTime() - r.days * 864e5);
      return api<History>("/history", { location, metric, resolution: r.resolution, start: start.toISOString(), end: end.toISOString() });
    },
    [location, metric, range, anchor],
  );

  const rows: Row[] = useMemo(
    () =>
      (data?.points ?? []).map((p) => {
        const ts = new Date(p.t.length === 10 ? `${p.t}T00:00:00+05:30` : p.t).getTime();
        return {
          label: p.t,
          ts,
          v: p.v,
          range: p.min != null && p.max != null ? [p.min, p.max] : null,
        };
      }),
    [data],
  );
  const yMax = Math.max(50, Math.ceil(Math.max(0, ...rows.map((x) => x.range?.[1] ?? x.v ?? 0)) * 1.05 / 50) * 50);
  const fmtTick = (ms: number) =>
    r.resolution === "monthly" ? ist(new Date(ms).toISOString(), { month: "short", year: "2-digit" }) : ist(new Date(ms).toISOString(), { day: "numeric", month: "short" });

  return (
    <section className="card" aria-labelledby="hist-title">
      <div className="card-head">
        <h2 id="hist-title">Historical trend</h2>
        <span className="sub">{r.resolution === "hourly" ? "Hourly values" : r.resolution === "daily" ? "Daily mean with min to max range" : "Monthly mean"} · Delhi local days</span>
        <div className="right">
          <div className="seg" role="group" aria-label="Metric">
            {METRICS.map((m) => (
              <button key={m.key} aria-pressed={metric === m.key} onClick={() => setMetric(m.key)}>{m.label}</button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Time range">
            {RANGES.map((x) => (
              <button key={x.key} aria-pressed={range === x.key} onClick={() => setRange(x.key)}>{x.label}</button>
            ))}
          </div>
        </div>
      </div>
      {error && <div className="err">Could not load history: {error}</div>}
      {!error && (loading && !data ? <div className="skeleton" /> : (
        <div style={{ width: "100%", height: 300, opacity: loading ? 0.6 : 1 }}>
          <ResponsiveContainer>
            <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              {metric === "aqi" && categories.map((c) => (
                <ReferenceArea key={c.name} y1={c.low} y2={Math.min(c.high, yMax)} fill={c.color} fillOpacity={0.1} ifOverflow="hidden" />
              ))}
              <XAxis dataKey="ts" type="number" scale="time" domain={["dataMin", "dataMax"]} tickCount={7} tickFormatter={fmtTick} stroke="var(--muted)" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, yMax]} width={44} stroke="var(--muted)" tick={{ fontSize: 12 }} />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const p = payload[0].payload as Row;
                  return (
                    <div className="tip">
                      <div>{r.resolution === "monthly" ? ist(new Date(p.ts).toISOString(), { month: "long", year: "numeric" }) : p.label.length === 10 ? ist(new Date(p.ts).toISOString(), { weekday: "short", day: "numeric", month: "short", year: "numeric" }) : ist(p.label, { day: "numeric", month: "short", hour: "numeric", hour12: true })}</div>
                      <div>{METRICS.find((m) => m.key === metric)?.label}: <b>{fixed(p.v, 0)}</b> {data?.unit}</div>
                      {p.range && <div>range: {fixed(p.range[0], 0)} to {fixed(p.range[1], 0)}</div>}
                    </div>
                  );
                }}
              />
              <Area dataKey="range" stroke="none" fill="var(--accent)" fillOpacity={0.15} isAnimationActive={false} connectNulls />
              <Line dataKey="v" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      ))}
    </section>
  );
}
