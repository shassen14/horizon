"use client";

import { useState } from "react";
import { MarketLeadersParams, MarketLeadersSortBy } from "@/lib/api";
import { MarketLeadersTable } from "@/components/market/MarketLeadersTable";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SlidersHorizontal, ArrowUpDown } from "lucide-react";

// --- Helpers for inputs (Reuse these) ---
const formatNumber = (num: number): string => {
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(0)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(0)}K`;
  return num.toString();
};

const parseFormattedNumber = (str: string): number => {
  const upper = str.toUpperCase();
  if (upper.endsWith("M"))
    return parseFloat(upper.replace("M", "")) * 1_000_000;
  if (upper.endsWith("K")) return parseFloat(upper.replace("K", "")) * 1_000;
  return parseFloat(str) || 0;
};

export function MarketScreener() {
  // 1. State Management
  // Default to a larger list for the dedicated page
  const [filters, setFilters] = useState<MarketLeadersParams>({
    sortBy: "relative_volume",
    sortDir: "desc",
    minPrice: 2.0,
    minAvgVolume: 500_000,
    limit: 100,
  });

  // Display state for volume input
  const [volumeDisplay, setVolumeDisplay] = useState(
    formatNumber(filters.minAvgVolume)
  );

  // 2. Handlers
  // "K" is the specific key (e.g., 'minPrice')
  // "value" must match the type of that key (e.g., number)
  const handleFilterChange = <K extends keyof MarketLeadersParams>(
    key: K,
    value: MarketLeadersParams[K]
  ) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const handleVolumeBlur = () => {
    const parsed = parseFormattedNumber(volumeDisplay);
    setFilters((prev) => ({ ...prev, minAvgVolume: parsed }));
    setVolumeDisplay(formatNumber(parsed));
  };

  const toggleSortDir = () => {
    setFilters((prev) => ({
      ...prev,
      sortDir: prev.sortDir === "asc" ? "desc" : "asc",
    }));
  };

  return (
    <div className="space-y-6">
      {/* --- Filter Control Panel --- */}
      <div className="bg-white p-4 rounded-lg border shadow-sm space-y-4">
        <div className="flex items-center gap-2 mb-2 text-slate-700">
          <SlidersHorizontal className="h-4 w-4" />
          <h3 className="font-semibold text-sm">Screener Settings</h3>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {/* Sort Column */}
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Sort By</Label>
            <Select
              value={filters.sortBy}
              onValueChange={(val) =>
                handleFilterChange("sortBy", val as MarketLeadersSortBy)
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="relative_volume">Relative Volume</SelectItem>
                <SelectItem value="return_1d">Daily Change</SelectItem>
                <SelectItem value="rsi_14">RSI</SelectItem>
                <SelectItem value="atr_14_pct">Volatility (ATR)</SelectItem>
                <SelectItem value="volume_adv_20">Avg Volume</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Sort Direction */}
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Direction</Label>
            <Button
              variant="outline"
              className="w-full justify-between"
              onClick={toggleSortDir}
            >
              {filters.sortDir === "desc" ? "Highest First" : "Lowest First"}
              <ArrowUpDown className="h-3 w-3 ml-2 opacity-50" />
            </Button>
          </div>

          {/* Min Price */}
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">
              Min Price ($)
            </Label>
            <Input
              type="number"
              value={filters.minPrice}
              onChange={(e) =>
                handleFilterChange("minPrice", parseFloat(e.target.value) || 0)
              }
            />
          </div>

          {/* Min Volume */}
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">
              Min Avg Volume
            </Label>
            <Input
              value={volumeDisplay}
              onChange={(e) => setVolumeDisplay(e.target.value)}
              onBlur={handleVolumeBlur}
            />
          </div>

          {/* Limit */}
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Max Results</Label>
            <Select
              value={filters.limit.toString()}
              onValueChange={(val) =>
                handleFilterChange("limit", parseInt(val))
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="50">50</SelectItem>
                <SelectItem value="100">100</SelectItem>
                <SelectItem value="200">200 (Max)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* --- Results Table --- */}
      {/* We reuse the dumb component here, passing our state down */}
      <MarketLeadersTable filters={filters} />
    </div>
  );
}
