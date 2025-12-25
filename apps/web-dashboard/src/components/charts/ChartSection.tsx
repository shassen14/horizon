// apps/web-dashboard/src/components/charts/ChartSection.tsx

"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getHistory, MarketInterval, HistoryDataPoint } from "@/lib/api";
import { ChartToolbar } from "@/components/charts/ChartToolbar"; // Adjusted import path based on previous moves
import { FinancialChart } from "@/components/charts/FinancialChart";
import { ChartControls } from "@/components/charts/ChartControls";
import { getInitialIndicatorState } from "@/config/indicators";
import { TimeRange, getDateRange } from "@/lib/date-ranges";
import { UTCTimestamp } from "lightweight-charts";

interface ChartSectionProps {
  symbol: string;
  initialData: HistoryDataPoint[];
}

// --- CONFIG: Define the Auto-Switch Logic ---
const INTERVAL_MAPPING: Record<TimeRange, MarketInterval> = {
  "1D": "5m",
  "1W": "1h",
  "1M": "1d",
  "3M": "1d",
  YTD: "1d",
  "1Y": "1d",
  "5Y": "1w", // or "1w" if you prefer less noise
  ALL: "1w",
};

export function ChartSection({ symbol, initialData }: ChartSectionProps) {
  const [interval, setInterval] = useState<MarketInterval>("1d");
  const [timeRange, setTimeRange] = useState<TimeRange>("1Y");
  const [selectedIndicators, setSelectedIndicators] = useState(
    getInitialIndicatorState()
  );

  const { startDate, endDate } = getDateRange(timeRange);

  const { data: historyData, isLoading } = useQuery({
    queryKey: ["marketHistory", symbol, interval, timeRange],
    queryFn: () => getHistory(symbol, 5000, interval, startDate, endDate),
    initialData:
      interval === "1d" && timeRange === "1Y" ? initialData : undefined,
    staleTime: 1000 * 60,
  });

  // --- THE FIX: Smart Handler ---
  const handleTimeRangeChange = (newRange: TimeRange) => {
    setTimeRange(newRange);

    // Automatically switch the interval to the best fit
    const bestInterval = INTERVAL_MAPPING[newRange];
    if (bestInterval) {
      setInterval(bestInterval);
    }
  };

  const dataToRender = historyData || initialData || [];

  if (dataToRender.length === 0) {
    return (
      <div className="p-4 text-center">
        No chart data available for this range.
      </div>
    );
  }

  const ohlcData = dataToRender.map((d) => ({
    time: (new Date(d.time).getTime() / 1000) as UTCTimestamp,
    open: d.open,
    high: d.high,
    low: d.low,
    close: d.close,
  }));

  const volumeData = dataToRender.map((d) => ({
    time: (new Date(d.time).getTime() / 1000) as UTCTimestamp,
    value: d.volume,
  }));

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-4 border rounded-lg p-4 bg-white shadow-sm">
        <div className="border-b pb-4">
          <ChartToolbar
            // interval={interval}
            // onIntervalChange={setInterval} // User can still manually override interval
            timeRange={timeRange}
            onTimeRangeChange={handleTimeRangeChange} // Use our smart handler here
          />
          <ChartControls
            selection={selectedIndicators}
            onSelectionChange={setSelectedIndicators}
          />
        </div>

        <div
          className={`transition-opacity duration-200 ${
            isLoading ? "opacity-50" : "opacity-100"
          }`}
        >
          <FinancialChart
            ohlcData={ohlcData}
            volumeData={volumeData}
            technicalsData={dataToRender}
            selectedIndicators={selectedIndicators}
          />
        </div>
      </div>
    </div>
  );
}
