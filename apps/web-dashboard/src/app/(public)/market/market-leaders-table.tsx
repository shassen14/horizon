// apps/web-dashboard/src/app/(public)/market/market-leaders-table.tsx

"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getMarketLeaders, MarketLeadersParams } from "@/lib/api";
import Link from "next/link";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Terminal } from "lucide-react";

// A helper to format large numbers
const formatNumber = (num: number): string => {
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(0)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(0)}K`;
  return num.toString();
};

// A helper to parse formatted strings back to numbers
const parseFormattedNumber = (str: string): number => {
  const upper = str.toUpperCase();
  if (upper.endsWith("M")) {
    return parseFloat(upper.replace("M", "")) * 1_000_000;
  }
  if (upper.endsWith("K")) {
    return parseFloat(upper.replace("K", "")) * 1_000;
  }
  return parseFloat(str) || 0;
};

export function MarketLeadersTable() {
  // State for the entire filter form
  const [filters, setFilters] = useState<MarketLeadersParams>({
    sortBy: "relative_volume",
    minPrice: 10.0,
    minAvgVolume: 2_000_000,
    limit: 50,
  });

  const [volumeDisplay, setVolumeDisplay] = useState(
    formatNumber(filters.minAvgVolume)
  );

  // This state holds the filters that are actually *applied*.
  // It only changes when the user clicks the button.
  const [appliedFilters, setAppliedFilters] = useState(filters);

  // useQuery now depends on 'appliedFilters'. It will automatically refetch
  // when this state changes.
  const {
    data: leaders,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["marketLeaders", appliedFilters],
    queryFn: () => getMarketLeaders(appliedFilters),
    // It runs automatically on mount, no need for manual refetch.
  });

  // The input fields are now controlled by the 'filters' state
  const handleApplyFilters = () => {
    // Before applying, ensure the raw filter state is updated from the display state
    const parsedVolume = parseFormattedNumber(volumeDisplay);
    const finalFilters = { ...filters, minAvgVolume: parsedVolume };

    setFilters(finalFilters); // Update the main state
    setAppliedFilters(finalFilters); // Trigger the query
  };

  const handleVolumeBlur = () => {
    // When user clicks away, parse the input and reformat it
    const parsed = parseFormattedNumber(volumeDisplay);
    setFilters((prev) => ({ ...prev, minAvgVolume: parsed }));
    setVolumeDisplay(formatNumber(parsed));
  };

  return (
    <div className="space-y-6">
      {/* --- Filter Form --- */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 p-4 border rounded-lg">
        <div className="space-y-2">
          <Label htmlFor="sort-by">Sort By</Label>
          <Select
            value={filters.sortBy}
            onValueChange={(value) =>
              setFilters((prev) => ({ ...prev, sortBy: value as any }))
            }
          >
            <SelectTrigger id="sort-by">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="relative_volume">Relative Volume</SelectItem>
              <SelectItem value="return_1d">Daily Change (%)</SelectItem>
              <SelectItem value="rsi_14">RSI (14)</SelectItem>
              <SelectItem value="atr_14_pct">ATR (%)</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label htmlFor="min-price">Min Price</Label>
          <Input
            id="min-price"
            type="number"
            value={filters.minPrice}
            onChange={(e) =>
              setFilters((prev) => ({
                ...prev,
                minPrice: parseFloat(e.target.value) || 0,
              }))
            }
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="min-volume">Min Avg Volume ($)</Label>
          <Input
            id="min-volume"
            type="text" // Change to text to allow "2M"
            value={volumeDisplay}
            onChange={(e) => setVolumeDisplay(e.target.value)}
            onBlur={handleVolumeBlur} // Update state on blur
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="limit">Results</Label>
          <Input
            id="limit"
            type="number"
            value={filters.limit}
            onChange={(e) =>
              setFilters((prev) => ({
                ...prev,
                limit: parseInt(e.target.value) || 20,
              }))
            }
          />
        </div>
        <div className="col-span-2 flex items-end">
          <Button onClick={handleApplyFilters} className="w-full">
            {isLoading ? "Loading..." : "Apply Filters"}
          </Button>
        </div>
      </div>

      {/* --- Results Section --- */}
      <div>
        {isLoading && (
          <div className="space-y-2">
            {[...Array(10)].map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        )}
        {isError && (
          <Alert variant="destructive">
            <Terminal className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>
              Failed to load market data. The backend may be offline.
            </AlertDescription>
          </Alert>
        )}
        {!isLoading && !isError && leaders && (
          <div className="border rounded-lg">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[60px]">Rank</TableHead>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Price</TableHead>
                  <TableHead>Change (%)</TableHead>
                  <TableHead>Rel. Volume</TableHead>
                  <TableHead>RSI (14)</TableHead>
                  <TableHead>Dist from SMA50</TableHead>
                  <TableHead>Volatility (ATR%)</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {leaders.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} className="h-24 text-center">
                      No stocks match your criteria.
                    </TableCell>
                  </TableRow>
                ) : (
                  leaders.map((stock) => (
                    <TableRow key={stock.symbol}>
                      <TableCell className="font-medium">
                        {stock.rank}
                      </TableCell>
                      <TableCell>
                        <Link
                          href={`/market/${stock.symbol}`}
                          className="font-bold hover:underline"
                        >
                          {stock.symbol}
                        </Link>
                        <p className="text-xs text-muted-foreground w-[150px] truncate">
                          {stock.name}
                        </p>
                      </TableCell>
                      <TableCell>${stock.latest_price.toFixed(2)}</TableCell>
                      <TableCell
                        className={
                          stock.daily_change_pct >= 0
                            ? "text-green-600"
                            : "text-red-600"
                        }
                      >
                        {stock.daily_change_pct.toFixed(2)}%
                      </TableCell>
                      <TableCell>
                        {stock.relative_volume?.toFixed(2)}x
                      </TableCell>
                      <TableCell>{stock.rsi_14?.toFixed(2)}</TableCell>
                      <TableCell>
                        {stock.sma_50_pct_diff
                          ? `${(stock.sma_50_pct_diff * 100).toFixed(1)}%`
                          : "N/A"}
                      </TableCell>
                      <TableCell>{stock.atr_14_pct?.toFixed(2)}%</TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        )}
      </div>
    </div>
  );
}
