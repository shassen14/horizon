// apps/web-dashboard/src/app/(public)/market/[symbol]/page.tsx

import { getHistory } from "@/lib/api";
import { FeatureTable } from "@/components/business/FeatureTable";
import { notFound } from "next/navigation";
import { ChartSection } from './chart-section';

// This is the stable, correct signature for Next.js 14
export default async function StockDetailPage({ params }: { params: { symbol: string } }) {
  const symbol = params.symbol.toUpperCase();
  
  // Fetch data on the server
  const initialData = await getHistory(symbol, 365 * 2); 

  // If API returns no data (e.g., 404), show the Not Found page.
  if (!initialData || initialData.length === 0) {
    notFound();
  }

  const reversedData = [...initialData].reverse();

  // Prepare data for the chart component
  const ohlcData = reversedData.map(d => ({
    time: (new Date(d.time).getTime() / 1000) as any,
    open: d.open,
    high: d.high,
    low: d.low,
    close: d.close,
  }));

  const volumeData = reversedData.map(d => ({
    time: (new Date(d.time).getTime() / 1000) as any,
    value: d.volume,
  }));

  // The API returns data sorted by latest first, so index 0 is the newest.
  const latestFeatures = initialData[0]?.features;

  return (
    <div className="container mx-auto p-4 space-y-8">
      <div>
        <h1 className="text-3xl font-bold">{symbol}</h1>
        <p className="text-muted-foreground">Daily Chart & Analysis</p>
      </div>
        <ChartSection 
          ohlcData={ohlcData} 
          volumeData={volumeData} 
          technicalsData={initialData} // Pass un-reversed for indicator logic
        />
      <div>
        <h2 className="text-2xl font-bold">Latest Technical Features</h2>
        <p className="text-muted-foreground">
          As of {new Date(initialData[0].time).toLocaleDateString()}
        </p>
      </div>

      {latestFeatures ? (
        <FeatureTable features={latestFeatures} />
      ) : (
        <p>No feature data available.</p>
      )}
    </div>
  );
}