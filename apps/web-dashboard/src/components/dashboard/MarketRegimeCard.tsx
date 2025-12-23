// apps/web-dashboard/src/components/dashboard/MarketRegimeCard.tsx

import { MarketRegime, RegimeType, RiskLevel } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  Rocket,
  Eye,
  Shield,
  TrendingUp,
  TrendingDown,
  ArrowLeftRight,
} from "lucide-react";

// --- 1. Define the Mapping ---

// Map the abstract RiskLevel to a concrete Stance
const stanceConfig: Record<
  RiskLevel,
  {
    label: string;
    color: string;
    icon: React.ReactNode;
  }
> = {
  "Risk On": {
    label: "Aggressive",
    color: "bg-green-100 text-green-800",
    icon: <Rocket className="mr-1 h-3 w-3" />,
  },
  Neutral: {
    label: "Selective",
    color: "bg-yellow-100 text-yellow-800",
    icon: <Eye className="mr-1 h-3 w-3" />,
  },
  "Risk Off": {
    label: "Defensive",
    color: "bg-red-100 text-red-800",
    icon: <Shield className="mr-1 h-3 w-3" />,
  },
};

// Map the RegimeType to a color and icon
const regimeConfig: Record<
  RegimeType,
  {
    color: string;
    icon: React.ReactNode;
  }
> = {
  Bull: {
    color: "text-green-600",
    icon: <TrendingUp className="mr-2 h-5 w-5" />,
  },
  Bear: {
    color: "text-red-600",
    icon: <TrendingDown className="mr-2 h-5 w-5" />,
  },
  Sideways: {
    color: "text-slate-500",
    icon: <ArrowLeftRight className="mr-2 h-5 w-5" />,
  },
};

export function MarketRegimeCard({ regime }: { regime: MarketRegime | null }) {
  if (!regime) return null;

  // --- 2. Get the config for the current state ---
  const currentStance = stanceConfig[regime.risk_signal];
  const currentRegime = regimeConfig[regime.regime];

  const formatPercent = (val: number | null | undefined) =>
    val ? `${(val * 100).toFixed(1)}%` : "N/A";

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Market Regime Analysis</span>
          {/* The Stance Badge */}
          <Badge className={cn("border-none", currentStance.color)}>
            {currentStance.icon}
            {currentStance.label}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* The Regime Title */}
        <div
          className={cn(
            "flex items-center text-2xl font-bold",
            currentRegime.color
          )}
        >
          {currentRegime.icon}
          <span>{regime.regime} Trend</span>
        </div>
        {/* Supporting Metrics */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="space-y-1">
            <p className="text-muted-foreground">Market Breadth</p>
            <p className="font-semibold">{formatPercent(regime.breadth_pct)}</p>
            <p className="text-xs text-muted-foreground">
              (% stocks &gt SMA50)
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-muted-foreground">Avg. Volatility</p>
            <p className="font-semibold">
              {regime.market_volatility_avg?.toFixed(2)}%
            </p>
            <p className="text-xs text-muted-foreground">(Daily ATR%)</p>
          </div>
        </div>
        {/* The Summary */}
        <p className="text-sm text-muted-foreground">{regime.summary}</p>
      </CardContent>
    </Card>
  );
}
