// apps/web-dashboard/src/app/(public)/page.tsx

import {
  getSystemStatus,
  getMarketRegime,
  getMarketSnapshots,
} from "@/lib/api";
import { SystemStatusCard } from "@/components/dashboard/SystemStatusCard";
import { MarketRegimeCard } from "@/components/dashboard/MarketRegimeCard";
import { MarketSnapshotCard } from "@/components/dashboard/MarketSnapshotCard";
import { TopMoversSection } from "@/components/dashboard/TopMoversSection";

export default async function HomePage() {
  // Fetch all data in parallel for a fast page load
  const [systemStatus, marketRegime, marketSnapshots] = await Promise.all([
    getSystemStatus(),
    getMarketRegime(),
    getMarketSnapshots(["SPY", "QQQ", "IWM"]),
  ]);

  return (
    <div className="container mx-auto p-4 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Market Dashboard</h1>
        <p className="text-muted-foreground">
          An overview of system health, market conditions, and top movers.
        </p>
      </div>

      {/* Top Row: System & Market Context */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <MarketRegimeCard regime={marketRegime} />
        </div>
        <SystemStatusCard status={systemStatus} />
      </div>

      {/* Middle Row: Market Snapshots */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {marketSnapshots.map((snapshot) => (
          <MarketSnapshotCard key={snapshot.symbol} snapshot={snapshot} />
        ))}
      </div>

      {/* Bottom Section: Top Movers (Action) */}
      <div>
        <div className="mt-4">
          <TopMoversSection />
        </div>
      </div>
    </div>
  );
}
