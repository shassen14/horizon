// apps/web-dashboard/src/components/dashboard/MarketSnapshotCard.tsx

import { MarketSnapshot } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";

// Dynamically import the SparklineChart with SSR disabled.
// This tells Next.js to render a fallback (the Skeleton) on the server,
// and then load and render the actual chart component on the client.
const SparklineChart = dynamic(
  () => import("./SparklineChart").then((mod) => mod.SparklineChart),
  {
    ssr: false,
    loading: () => <Skeleton className="h-full w-full" />,
  }
);

export function MarketSnapshotCard({ snapshot }: { snapshot: MarketSnapshot }) {
  // 1. Capture the raw value (nullable)
  const changePct = snapshot.change_1d_pct;

  // 2. Determine if we actually have data
  const hasData = changePct !== null && changePct !== undefined;

  // 3. Determine Colors based on state
  let colorClass = "text-muted-foreground"; // Default: Gray/Neutral
  let sparklineColor = "#94a3b8"; // Default: Slate-400 (Gray)

  if (hasData) {
    const isUp = changePct >= 0;
    colorClass = isUp ? "text-green-600" : "text-red-600";
    sparklineColor = isUp ? "#16a34a" : "#dc2626";
  }

  const sparklineData = snapshot.sparkline.map((price, index) => ({
    name: index,
    value: price,
  }));

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-lg font-medium">{snapshot.symbol}</CardTitle>

        {/* The chart container still needs a defined size */}
        <div className="h-10 w-24">
          <SparklineChart data={sparklineData} color={sparklineColor} />
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">${snapshot.price.toFixed(2)}</div>
        <p className={cn("text-xs font-medium", colorClass)}>
          {hasData ? (
            // State A: We have data
            <>
              {changePct >= 0 ? "+" : ""}
              {(changePct * 100).toFixed(2)}% Today
            </>
          ) : (
            // State B: Missing data (The honest UI)
            <span className="text-muted-foreground">--</span>
          )}
        </p>
      </CardContent>
    </Card>
  );
}
