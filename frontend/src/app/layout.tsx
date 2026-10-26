import type { Metadata } from "next";
import { Nav } from "@/components/nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Demand Forecasting & Stockout Simulation",
  description: "Forecast 28-day retail demand, compare models, and simulate stockout risk (M5 subset).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <Nav />
        <main className="mx-auto max-w-[1280px] px-6 py-6">{children}</main>
        <footer className="mx-auto max-w-[1280px] px-6 pb-10 text-[12.5px] text-muted">
          Data: M5 Forecasting (Walmart) — 3 stores × top 100 items. Forecasts and metrics are precomputed;
          simulation outputs are model-based estimates, not measured history.
        </footer>
      </body>
    </html>
  );
}
