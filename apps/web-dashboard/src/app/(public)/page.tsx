// apps/web-dashboard/src/app/(public)/page.tsx

// This tells Next.js to opt out of Static Generation for this page.
// It resolves the "Dynamic server usage" build error.
export const dynamic = "force-dynamic";

import {
  getSystemStatus,
  getMarketRegime,
  getMarketSnapshots,
} from "@/lib/api";
import { SystemStatusCard } from "@/components/dashboard/SystemStatusCard";
import { MarketRegimeCard } from "@/components/dashboard/MarketRegimeCard";
import { MarketSnapshotCard } from "@/components/dashboard/MarketSnapshotCard";
import { TopMoversSection } from "@/components/dashboard/TopMoversSection";
import { WatchlistCard } from "@/components/dashboard/WatchlistCard";

export default async function HomePage() {
  // Fetch all data in parallel for a fast page load
  const [systemStatus, marketRegime, marketSnapshots] = await Promise.all([
    getSystemStatus(),
    getMarketRegime(),
    getMarketSnapshots(["SPY", "QQQ", "IWM", "VOO"]),
  ]);

  return (
    <div className="container mx-auto p-4 space-y-6">
      {/* 1. Header: Page Title */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Market Dashboard</h1>
        <p className="text-muted-foreground">
          Overview of system health and market conditions.
        </p>
      </div>

      {/* 2. Key Indices Row (Full Width, 4 Columns) */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {marketSnapshots.map((snapshot) => (
          <MarketSnapshotCard key={snapshot.symbol} snapshot={snapshot} />
        ))}
      </div>

      {/* 3. Context Row (Regime + System) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <MarketRegimeCard regime={marketRegime} />
        </div>
        <div className="lg:col-span-1">
          <SystemStatusCard status={systemStatus} />
        </div>
      </div>

      {/* 4. The Workspace Row (Movers + Watchlist) */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
        {/* Left: Top Movers (Takes up 75% width) */}
        <div className="lg:col-span-3">
          <TopMoversSection />
        </div>

        {/* Right: Watchlist Sidebar (Takes up 25% width) */}
        {/* Sticky keeps it accessible while you scroll the long table */}
        <div className="lg:col-span-1 sticky top-20">
          <WatchlistCard />
        </div>
      </div>
    </div>
  );
}
