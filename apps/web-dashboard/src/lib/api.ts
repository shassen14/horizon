// apps/web-dashboard/src/lib/api.ts

// We can auto-generate these types from OpenAPI later, but for now, we'll define them manually.

export interface TrendFeatures {
  sma_20: number | null;
  sma_50: number | null;
  sma_200: number | null;
  ema_12: number | null;
  ema_20: number | null;
  ema_26: number | null;
  ema_50: number | null;
  macd: number | null;
  macd_signal: number | null;
  macd_hist: number | null;
}

export interface MomentumFeatures {
  rsi_14: number | null;
  return_1d: number | null;
  return_5d: number | null;
  return_21d: number | null;
  return_63d: number | null;
}

export interface VolatilityFeatures {
  atr_14: number | null;
  atr_14_pct: number | null;
}

export interface VolumeFeatures { 
  volume_adv_20: number | null;
  relative_volume: number | null;
}

export interface FeatureSet {
  trend: TrendFeatures;
    momentum: MomentumFeatures;
    volatility: VolatilityFeatures;
    volume: VolumeFeatures;
}

export interface HistoryDataPoint {
  time: string; // Comes as an ISO string
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  features: FeatureSet;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function getHistory(symbol: string, limit: number = 1000): Promise<HistoryDataPoint[]> {
  const url = `${API_BASE_URL}/api/v1/public/market/history/${symbol}?limit=${limit}`;
  console.log(`Fetching history from: ${url}`);
  const res = await fetch(url, {
    // Revalidate data every 5 minutes
    next: { revalidate: 300 },
  });

  if (!res.ok) {
    // This will activate the closest `error.js` Error Boundary
    throw new Error(`Failed to fetch data for ${symbol}`);
  }

  return res.json();
}