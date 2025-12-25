// apps/web-dashboard/src/components/charts/ChartToolbar.tsx

"use client";

// import { MarketInterval } from "@/lib/api";
import { TimeRange } from "@/lib/date-ranges";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface ChartToolbarProps {
  //   interval: MarketInterval;
  //   onIntervalChange: (val: MarketInterval) => void;
  timeRange: TimeRange;
  onTimeRangeChange: (val: TimeRange) => void;
}

export function ChartToolbar({
  //   interval,
  //   onIntervalChange,
  timeRange,
  onTimeRangeChange,
}: ChartToolbarProps) {
  return (
    <div className="flex flex-col md:flex-row justify-between items-center gap-4 mb-4 w-full">
      {/* Left Side: Time Range (1D, 1W, 1M...) */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground mr-1">
          Range:
        </span>
        <Tabs
          value={timeRange}
          onValueChange={(v) => onTimeRangeChange(v as TimeRange)}
        >
          <TabsList className="h-8">
            {["1D", "1W", "1M", "3M", "YTD", "1Y", "5Y"].map((r) => (
              <TabsTrigger key={r} value={r} className="text-xs px-3 h-6">
                {r}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      {/* Right Side: Interval (5m, 1h, 1d...) */}
      {/* <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground mr-1">
          Interval:
        </span>
        <Tabs
          value={interval}
          onValueChange={(v) => onIntervalChange(v as MarketInterval)}
        >
          <TabsList className="h-8">
            <TabsTrigger value="5m" className="text-xs px-3 h-6">
              5m
            </TabsTrigger>
            <TabsTrigger value="15m" className="text-xs px-3 h-6">
              15m
            </TabsTrigger>
            <TabsTrigger value="1h" className="text-xs px-3 h-6">
              1h
            </TabsTrigger>
            <TabsTrigger value="4h" className="text-xs px-3 h-6">
              4h
            </TabsTrigger>
            <TabsTrigger value="1d" className="text-xs px-3 h-6">
              1D
            </TabsTrigger>
            <TabsTrigger value="1w" className="text-xs px-3 h-6">
              1W
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div> */}
    </div>
  );
}
