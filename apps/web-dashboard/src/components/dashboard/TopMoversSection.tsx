"use client";

import { useState } from "react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { MarketLeadersTable } from "@/components/market/MarketLeadersTable";
import { MarketLeadersParams } from "@/lib/api";

// Helpers for formatted inputs (2M -> 2000000)
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

type TabMode = "volume" | "gainers" | "losers";

export function TopMoversSection() {
  const [mode, setMode] = useState<TabMode>("volume");

  // Base State
  const [filters, setFilters] = useState<MarketLeadersParams>({
    sortBy: "relative_volume", // Default for 'volume' mode
    minPrice: 5.0,
    minAvgVolume: 2_000_000,
    limit: 10, // Show top 10 on dashboard
  });

  // Display State for Inputs
  const [volumeDisplay, setVolumeDisplay] = useState(
    formatNumber(filters.minAvgVolume)
  );

  // --- Logic to switch modes ---
  const handleTabChange = (value: string) => {
    const newMode = value as TabMode;
    setMode(newMode);

    // Apply presets based on tab
    if (newMode === "volume") {
      setFilters((prev) => ({
        ...prev,
        sortBy: "relative_volume",
        sortDir: "desc",
      }));
    } else if (newMode === "gainers") {
      setFilters((prev) => ({ ...prev, sortBy: "return_1d", sortDir: "desc" }));
    } else if (newMode === "losers") {
      setFilters((prev) => ({ ...prev, sortBy: "return_1d", sortDir: "asc" }));
    }
  };

  // --- Input Handlers ---
  const handleVolumeBlur = () => {
    const parsed = parseFormattedNumber(volumeDisplay);
    setFilters((prev) => ({ ...prev, minAvgVolume: parsed }));
    setVolumeDisplay(formatNumber(parsed));
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">
            Today&apos;s Top Movers
          </h2>
          <p className="text-muted-foreground text-sm">
            Real-time market activity scan.
          </p>
        </div>

        {/* --- Unified Control Bar --- */}
        <div className="flex items-center space-x-4 bg-slate-50 p-1 rounded-lg border">
          {/* Tabs */}
          <Tabs value={mode} onValueChange={handleTabChange}>
            <TabsList className="h-9">
              <TabsTrigger value="volume">High Volume</TabsTrigger>
              <TabsTrigger value="gainers">Gainers</TabsTrigger>
              <TabsTrigger value="losers">Losers</TabsTrigger>
            </TabsList>
          </Tabs>

          <div className="w-px h-6 bg-slate-200 hidden md:block"></div>

          {/* Quick Filters Inline */}
          <div className="hidden md:flex items-center space-x-2">
            <div className="flex items-center space-x-2">
              <Label
                htmlFor="price"
                className="text-xs text-muted-foreground whitespace-nowrap"
              >
                Min Price $
              </Label>
              <Input
                id="price"
                type="number"
                className="h-8 w-16 text-right"
                value={filters.minPrice}
                onChange={(e) =>
                  setFilters((prev) => ({
                    ...prev,
                    minPrice: parseFloat(e.target.value) || 0,
                  }))
                }
              />
            </div>
            <div className="flex items-center space-x-2">
              <Label
                htmlFor="vol"
                className="text-xs text-muted-foreground whitespace-nowrap"
              >
                Min Vol
              </Label>
              <Input
                id="vol"
                className="h-8 w-16 text-right"
                value={volumeDisplay}
                onChange={(e) => setVolumeDisplay(e.target.value)}
                onBlur={handleVolumeBlur}
              />
            </div>
          </div>
        </div>
      </div>

      {/* --- The Table (controlled by props) --- */}
      <MarketLeadersTable filters={filters} />

      {/* Mobile-only filter hint */}
      <p className="md:hidden text-xs text-center text-muted-foreground">
        Filters: Min ${filters.minPrice}, Vol {volumeDisplay}+
      </p>
    </div>
  );
}
