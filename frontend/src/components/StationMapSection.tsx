"use client";

import dynamic from "next/dynamic";
import { api, type Stations } from "@/lib/api";
import { useApi } from "@/lib/useApi";

// Leaflet touches `window`, so it is loaded in the browser only.
const StationMap = dynamic(() => import("./StationMap"), { ssr: false, loading: () => <div className="skeleton" style={{ height: 460 }} /> });

export function StationMapSection() {
  const { data, error } = useApi<Stations>(() => api<Stations>("/stations"), [], 15 * 60 * 1000);
  return (
    <section className="card" aria-labelledby="map-title">
      <div className="card-head">
        <h2 id="map-title">Map</h2>
        <span className="sub">Neighbourhoods coloured by the AQI of their model cell</span>
      </div>
      {error && <div className="err">Could not load the map data: {error}</div>}
      {data && (
        <>
          <StationMap data={data} />
          <p className="note" style={{ marginTop: 10 }}>
            {data.note}{" "}
            {data.ground_stations_enabled
              ? `Black-ringed markers are real ground sensors from OpenAQ (${data.ground_stations.length} reporting now).`
              : "Ground sensors from OpenAQ are switched off: set OPENAQ_API_KEY to show them."}
          </p>
        </>
      )}
    </section>
  );
}
