// apps/web-dashboard/src/components/business/FeatureTable.tsx

import { FeatureSet } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface FeatureTableProps {
  features: FeatureSet;
}

export function FeatureTable({ features }: FeatureTableProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <Card>
        <CardHeader><CardTitle>Trend</CardTitle></CardHeader>
        <CardContent>
          <p>SMA 50: {features.trend.sma_50?.toFixed(2)}</p>
          <p>SMA 200: {features.trend.sma_200?.toFixed(2)}</p>
          <p>MACD Hist: {features.trend.macd_hist?.toFixed(2)}</p>
        </CardContent>
      </Card>
       {/* Add similar cards for Momentum, Volatility, and Volume */}
    </div>
  );
}