// apps/web-dashboard/src/components/dashboard/WatchlistCard.tsxs

"use client";

import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { useWatchlist } from "@/hooks/useWatchlist";
import { getMarketSnapshots } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area"; // Import this
import Link from "next/link";
import { ArrowUpRight, ArrowDownRight, Minus } from "lucide-react"; // Nice touch for icons
import { WatchlistButton } from "@/components/market/WatchlistButton";

export function WatchlistCard() {
  const { watchlist, isLoaded } = useWatchlist();

  const { data: snapshots } = useQuery({
    queryKey: ["watchlist", watchlist],
    queryFn: () => getMarketSnapshots(watchlist),
    enabled: isLoaded && watchlist.length > 0,
    refetchInterval: 60000, // Auto-refresh every minute
    // This tells React Query: "If the key changes (new stock added),
    // keep displaying the OLD list until the NEW list arrives."
    placeholderData: keepPreviousData,
  });

  if (!isLoaded) return <Skeleton className="h-[350px] w-full" />;

  // Sort watchlist alphabetically or by change? Let's keep it simple for now.
  // const sortedSnapshots = snapshots?.sort((a, b) => b.change_1d_pct - a.change_1d_pct);

  return (
    <Card className="h-full flex flex-col border-none shadow-none bg-transparent sm:border sm:shadow-sm sm:bg-card">
      <CardHeader className="pb-3 px-0 sm:px-6">
        <CardTitle className="text-base font-semibold">
          Your Watchlist
        </CardTitle>
      </CardHeader>

      <CardContent className="p-0 flex-1 min-h-0">
        {watchlist.length === 0 ? (
          <div className="p-6 text-center text-sm text-muted-foreground border-dashed border-2 rounded-lg m-4">
            No stocks starred yet.
            <br />
            <span className="text-xs">Search or browse to add some!</span>
          </div>
        ) : (
          <ScrollArea className="h-[350px] sm:px-4">
            <div className="flex flex-col gap-1 pb-4">
              {snapshots?.map((stock) => {
                const change = stock.change_1d_pct;
                const hasData = typeof change === "number"; // Strict number check

                const isUp = hasData ? change >= 0 : false;

                // Determine Colors based on state (Green, Red, or Neutral Gray)
                const badgeColor = hasData
                  ? isUp
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                  : "bg-slate-100 text-slate-500";

                const textColor = hasData
                  ? isUp
                    ? "text-green-600"
                    : "text-red-600"
                  : "text-muted-foreground";

                return (
                  <div key={stock.symbol} className="relative group">
                    <Link
                      href={`/market/${stock.symbol}`}
                      className="flex items-center justify-between p-3 rounded-lg hover:bg-slate-50 transition-all border border-transparent hover:border-slate-100 pr-10"
                    >
                      <div className="flex items-center gap-3">
                        {/* Trend Icon Badge */}
                        <div className={`p-2 rounded-full ${badgeColor}`}>
                          {hasData ? (
                            isUp ? (
                              <ArrowUpRight className="w-4 h-4" />
                            ) : (
                              <ArrowDownRight className="w-4 h-4" />
                            )
                          ) : (
                            <Minus className="w-4 h-4" /> // Neutral icon
                          )}
                        </div>

                        <div>
                          <div className="font-bold text-sm text-slate-900">
                            {stock.symbol}
                          </div>
                        </div>
                      </div>

                      <div className="text-right">
                        <div className="font-mono font-medium text-sm">
                          ${stock.price.toFixed(2)}
                        </div>
                        <div className={`text-xs font-medium ${textColor}`}>
                          {hasData ? (
                            <>
                              {isUp ? "+" : ""}
                              {(change * 100).toFixed(2)}%
                            </>
                          ) : (
                            "--"
                          )}
                        </div>
                      </div>
                    </Link>

                    <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <WatchlistButton symbol={stock.symbol} />
                    </div>
                  </div>
                );
              })}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}
