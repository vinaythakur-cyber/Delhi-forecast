import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Delhi AQI: air quality, 72-hour forecast and honest model evaluation",
  description:
    "Current Delhi air quality on the Indian National AQI scale, a 72-hour probabilistic forecast, historical trends and a transparent walk-forward model evaluation.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
