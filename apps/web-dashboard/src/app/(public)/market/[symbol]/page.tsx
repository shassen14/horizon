// apps/web-dashboard/src/app/(public)/market/[symbol]/page.tsx

import { getHistory, getAssetDetail } from "@/lib/api";
import { FeatureTable } from "@/components/business/FeatureTable";
import { ChartSection } from "@/components/charts/ChartSection";
import { KeyStatistics } from "@/components/market/KeyStatistics";
import { WatchlistButton } from "@/components/market/WatchlistButton";
import { notFound } from "next/navigation";

export default async function StockDetailPage({
  params,
}: {
  params: { symbol: string };
}) {
  const symbol = params.symbol.toUpperCase();

  // Fetch History AND Asset Details in parallel
  const [initialData, assetDetail] = await Promise.all([
    getHistory(symbol, 365, "1d"),
    getAssetDetail(symbol),
  ]);

  if (!assetDetail) {
    notFound();
  }

  // Handle case where history might be empty but asset exists
  const safeInitialData = initialData || [];
  const latestFeatures =
    safeInitialData.length > 0 ? safeInitialData[0].features : null;

  return (
    <div className="container mx-auto p-4 space-y-8">
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-3xl font-bold">{symbol}</h1>
          <WatchlistButton symbol={symbol} />
        </div>
        <p className="text-muted-foreground">{assetDetail.name}</p>
      </div>

      {/* 1. Interactive Chart Section */}
      <ChartSection symbol={symbol} initialData={safeInitialData} />

      {/* 2. Key Statistics */}
      <KeyStatistics asset={assetDetail} />

      {/* 3. Feature Data */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Latest Technical Analysis</h2>
        {latestFeatures ? (
          <FeatureTable features={latestFeatures} />
        ) : (
          <p>No technical data available.</p>
        )}
      </div>
    </div>
  );
}
