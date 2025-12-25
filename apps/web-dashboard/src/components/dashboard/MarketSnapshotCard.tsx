// apps/web-dashboard/src/components/dashboard/MarketSnapshotCard.tsx

"use client";

import { MarketSnapshot } from "@/lib/api";
import { Card } from "@/components/ui/card"; // Removing Header/Content for custom layout
import { cn } from "@/lib/utils";
import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";

const SparklineChart = dynamic(
  () => import("./SparklineChart").then((mod) => mod.SparklineChart),
  { ssr: false, loading: () => <Skeleton className="h-full w-full" /> }
);

export function MarketSnapshotCard({ snapshot }: { snapshot: MarketSnapshot }) {
  const changePct = snapshot.change_1d_pct;
  const hasData = changePct !== null && changePct !== undefined;

  let colorClass = "text-muted-foreground";
  let sparklineColor = "#94a3b8";

  if (hasData) {
    const isUp = changePct >= 0;
    colorClass = isUp ? "text-green-600" : "text-red-600";
    sparklineColor = isUp ? "#16a34a" : "#dc2626";
  }

  const sparklineData = snapshot.sparkline.map((price, index) => ({
    name: index,
    value: price,
  }));

  // --- COMPACT LAYOUT ---
  return (
    <Card className="p-4 flex flex-row items-center justify-between h-24">
      {/* Left: Symbol & Price */}
      <div className="flex flex-col justify-between h-full">
        <div className="text-sm font-semibold text-muted-foreground">
          {snapshot.symbol}
        </div>
        <div>
          <div className="text-xl font-bold">${snapshot.price.toFixed(2)}</div>
          <div className={cn("text-xs font-medium", colorClass)}>
            {hasData ? (
              <>
                {changePct >= 0 ? "+" : ""}
                {(changePct * 100).toFixed(2)}%
              </>
            ) : (
              "--"
            )}
          </div>
        </div>
      </div>

      {/* Right: Sparkline */}
      <div className="h-12 w-20">
        <SparklineChart data={sparklineData} color={sparklineColor} />
      </div>
    </Card>
  );
}
