"use client";

import "leaflet/dist/leaflet.css";
import { Circle, CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import type { Stations } from "@/lib/api";
import { istFull } from "@/lib/format";

// The two model cells are ~40 km wide in reality; a 14 km circle marks the point that was queried
// without pretending to show the cell's true outline.
export default function StationMap({ data }: { data: Stations }) {
  return (
    <MapContainer center={[28.64, 77.17]} zoom={10} scrollWheelZoom={false} style={{ height: 460, width: "100%" }}>
      <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {data.cells.map((c) => (
        <Circle key={c.id} center={[c.lat, c.lon]} radius={14000} pathOptions={{ color: c.color, weight: 2, fillColor: c.color, fillOpacity: 0.12, dashArray: "6 4" }}>
          <Popup>
            <b>{c.name}</b> (model cell)<br />AQI {c.aqi}, {c.category}<br />Data through {istFull(c.observed_at)}
          </Popup>
        </Circle>
      ))}
      {data.neighbourhoods.map((n) => (
        <CircleMarker key={n.name} center={[n.lat, n.lon]} radius={8} pathOptions={{ color: "#ffffff", weight: 2, fillColor: n.color, fillOpacity: 0.95 }}>
          <Popup>
            <b>{n.name}</b><br />AQI {n.aqi}, {n.category}<br />
            <small>Uses the {data.cells.find((c) => c.id === n.cell)?.name} model cell</small>
          </Popup>
        </CircleMarker>
      ))}
      {data.ground_stations.map((s) => (
        <CircleMarker key={s.id} center={[s.lat, s.lon]} radius={7} pathOptions={{ color: "#111", weight: 3, fillColor: s.color ?? "#888", fillOpacity: 1 }}>
          <Popup>
            <b>{s.name}</b> (ground sensor)<br />PM2.5 {s.pm2_5} µg/m³ → sub-index {s.pm2_5_sub_index}, {s.category}<br />
            <small>Latest hourly reading, {istFull(s.observed_at)}</small>
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
