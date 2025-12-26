// apps/web-dashboard/src/app/(public)/market/market-leaders-table.tsx

"use client";

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
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Terminal } from "lucide-react";
import { WatchlistButton } from "@/components/market/WatchlistButton";

interface MarketLeadersTableProps {
  filters: MarketLeadersParams;
}

export function MarketLeadersTable({ filters }: MarketLeadersTableProps) {
  const {
    data: leaders,
    isLoading,
    isError,
  } = useQuery({
    // The query key includes 'filters', so it auto-refetches when parent changes them
    queryKey: ["marketLeaders", filters],
    queryFn: () => getMarketLeaders(filters),
  });

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[...Array(5)].map((_, i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    );
  }

  if (isError) {
    return (
      <Alert variant="destructive">
        <Terminal className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>Failed to load market data.</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="border rounded-lg bg-white overflow-hidden">
      <Table>
        <TableHeader className="bg-slate-50">
          <TableRow>
            <TableHead className="w-[40px]"></TableHead>
            <TableHead className="w-[60px]">Rank</TableHead>
            <TableHead>Symbol</TableHead>
            <TableHead>Price</TableHead>
            <TableHead>Change (%)</TableHead>
            <TableHead>Rel. Vol</TableHead>
            <TableHead className="hidden md:table-cell">RSI (14)</TableHead>
            <TableHead className="hidden md:table-cell">Volatility</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {!leaders || leaders.length === 0 ? (
            <TableRow>
              <TableCell
                colSpan={7}
                className="h-24 text-center text-muted-foreground"
              >
                No stocks match your criteria.
              </TableCell>
            </TableRow>
          ) : (
            leaders.map((stock) => (
              <TableRow
                key={stock.symbol}
                className="hover:bg-slate-50/50 transition-colors"
              >
                <TableCell>
                  <WatchlistButton symbol={stock.symbol} />
                </TableCell>
                <TableCell className="font-medium text-slate-500">
                  #{stock.rank}
                </TableCell>
                <TableCell>
                  <Link
                    href={`/market/${stock.symbol}`}
                    className="font-bold hover:underline text-blue-600"
                  >
                    {stock.symbol}
                  </Link>
                  <p className="text-xs text-muted-foreground w-[120px] truncate">
                    {stock.name}
                  </p>
                </TableCell>
                <TableCell>${stock.latest_price.toFixed(2)}</TableCell>
                <TableCell>
                  <span
                    className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                      stock.daily_change_pct >= 0
                        ? "bg-green-100 text-green-800"
                        : "bg-red-100 text-red-800"
                    }`}
                  >
                    {stock.daily_change_pct >= 0 ? "+" : ""}
                    {stock.daily_change_pct.toFixed(2)}%
                  </span>
                </TableCell>
                <TableCell className="font-mono text-xs">
                  {stock.relative_volume
                    ? `${stock.relative_volume.toFixed(1)}x`
                    : "-"}
                </TableCell>
                <TableCell className="hidden md:table-cell">
                  {stock.rsi_14?.toFixed(0) ?? "-"}
                </TableCell>
                <TableCell className="hidden md:table-cell text-muted-foreground text-xs">
                  {stock.atr_14_pct
                    ? `${(stock.atr_14_pct * 100).toFixed(2)}%`
                    : "-"}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
