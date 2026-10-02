"use client";

import Link from "next/link";
import { ago, istFull } from "@/lib/format";
import type { Freshness } from "@/lib/api";

const LOCATIONS = [
  { id: "delhi", name: "Delhi (city average)" },
  { id: "delhi-north", name: "North & Central Delhi" },
  { id: "delhi-south", name: "South Delhi" },
];

interface Props {
  active: "dashboard" | "model";
  location?: string;
  onLocation?: (id: string) => void;
  freshness?: Freshness | null;
}

export function Header({ active, location, onLocation, freshness }: Props) {
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <Link href="/" className="brand">
          <span className="brand-mark" aria-hidden />
          Delhi AQI
        </Link>
        <nav className="nav" aria-label="Pages">
          <Link href="/" aria-current={active === "dashboard" ? "page" : undefined}>Dashboard</Link>
          <Link href="/model/" aria-current={active === "model" ? "page" : undefined}>Model performance</Link>
        </nav>
        <span className="spacer" />
        {onLocation && (
          <label>
            <span className="sr-only" style={{ position: "absolute", left: -9999 }}>Location</span>
            <select className="select" value={location} onChange={(e) => onLocation(e.target.value)} aria-label="Location">
              {LOCATIONS.map((l) => (
                <option key={l.id} value={l.id}>{l.name}</option>
              ))}
            </select>
          </label>
        )}
        {freshness && (
          <span
            className={`chip ${freshness.stale ? "stale" : "fresh"}`}
            title={`Latest data: ${istFull(freshness.data_through)}. ${freshness.note}`}
          >
            Data through {istFull(freshness.data_through)} · {ago(freshness.age_minutes)}
          </span>
        )}
      </div>
    </header>
  );
}
