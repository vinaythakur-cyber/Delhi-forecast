"use client";

import { useEffect, useState } from "react";
import { AqiCard } from "@/components/AqiCard";
import { ForecastChart } from "@/components/ForecastChart";
import { Header } from "@/components/Header";
import { HistoryChart } from "@/components/HistoryChart";
import { SeasonalHeatmap } from "@/components/SeasonalHeatmap";
import { StationMapSection } from "@/components/StationMapSection";
import { api, type Current, type Forecast } from "@/lib/api";
import { useApi } from "@/lib/useApi";

const REFRESH_MS = 5 * 60 * 1000;

function SetupPanel() {
  return (
    <div className="panel" role="status">
      <h2 style={{ marginBottom: 8 }}>First-time setup is still running</h2>
      <p>
        The server is downloading about four years of hourly air-quality data and training the forecasting models.
        This takes a few minutes on the very first start. This page checks again every 15 seconds.
      </p>
    </div>
  );
}

export default function Dashboard() {
  const [location, setLocation] = useState("delhi");
  const [pollMs, setPollMs] = useState(REFRESH_MS);
  const current = useApi(() => api<Current>("/current", { location }), [location, pollMs], pollMs);
  const forecast = useApi(() => api<Forecast>("/forecast", { location }), [location, pollMs], pollMs);
  const initialising = current.status === 503 || forecast.status === 503;
  // While the server is still doing its first-time setup, poll quickly; afterwards every 5 minutes.
  useEffect(() => setPollMs(initialising ? 15000 : REFRESH_MS), [initialising]);

  return (
    <>
      <Header active="dashboard" location={location} onLocation={setLocation} freshness={current.data?.freshness} />
      <main className="shell">
        {initialising && !current.data ? (
          <div className="grid"><SetupPanel /></div>
        ) : (
          <>
            {current.data?.freshness.stale && (
              <p className="note warn" style={{ marginTop: 16 }} role="alert">
                The latest data is more than 3 hours old. The source may be delayed; values below are the most recent available.
              </p>
            )}
            <div className="grid two">
              {current.error && !current.data ? <div className="err" role="alert">Could not load current air quality: {current.error}</div>
                : current.data ? <AqiCard data={current.data} /> : <div className="card"><div className="skeleton" style={{ height: 420 }} /></div>}
              {forecast.error && !forecast.data ? <div className="err" role="alert">Could not load the forecast: {forecast.error}</div>
                : forecast.data ? <ForecastChart data={forecast.data} /> : <div className="card"><div className="skeleton" style={{ height: 420 }} /></div>}
            </div>
            <div className="grid">
              <HistoryChart location={location} anchor={current.data?.freshness.data_through ?? null} categories={current.data?.categories ?? []} />
              <SeasonalHeatmap location={location} categories={current.data?.categories ?? []} />
              <StationMapSection />
            </div>
          </>
        )}
        <footer>
          <p>
            Air-quality values are produced by the Copernicus Atmosphere Monitoring Service (CAMS) model and served by{" "}
            <a href="https://open-meteo.com/">Open-Meteo</a> (CC BY 4.0). They are model estimates on a coarse grid, not sensor readings, and the
            newest hours are revised as better data arrives. AQI uses the Indian National AQI breakpoints (CPCB). Not a substitute for official
            health advice. <a href="/model/">See how the forecast is evaluated.</a>
          </p>
        </footer>
      </main>
    </>
  );
}
