"use client";

import { Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useWatchlist } from "@/hooks/useWatchlist";
import { cn } from "@/lib/utils";
import React from "react";

interface WatchlistButtonProps {
  symbol: string;
  className?: string;
}

export function WatchlistButton({ symbol, className }: WatchlistButtonProps) {
  const { isInWatchlist, toggleSymbol, isLoaded } = useWatchlist();

  // Prevent hydration mismatch or layout shift
  if (!isLoaded) return <div className={cn("w-8 h-8", className)} />;

  const isActive = isInWatchlist(symbol);

  const handleClick = (e: React.MouseEvent) => {
    // CRITICAL: Stop the click from bubbling up to parent Links/Rows
    e.preventDefault();
    e.stopPropagation();
    toggleSymbol(symbol);
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={handleClick}
      className={cn(
        "h-8 w-8 hover:bg-transparent", // Default compact size
        isActive
          ? "text-yellow-400 hover:text-yellow-500"
          : "text-slate-300 hover:text-slate-400",
        className
      )}
    >
      <Star
        className={cn("h-5 w-5", isActive && "fill-yellow-400")}
        strokeWidth={isActive ? 0 : 2} // Solid fill when active, outline when not
      />
    </Button>
  );
}
