// apps/web-dashboard/src/lib/api.ts

import { components } from "./api-client"; // Import the auto-generated types

// System
export type SystemHealth = components["schemas"]["SystemHealth"];
export type Environment = components["schemas"]["Environment"];
export type SystemStatus = components["schemas"]["SystemStatus"];

// Market
export type HistoryDataPoint = components["schemas"]["HistoryDataPoint"];
export type FeatureSet = components["schemas"]["FeatureSet"];
export type TrendFeatures = components["schemas"]["TrendFeatures"];
export type MomentumFeatures = components["schemas"]["MomentumFeatures"];
export type VolatilityFeatures = components["schemas"]["VolatilityFeatures"];
export type VolumeFeatures = components["schemas"]["VolumeFeatures"];
export type MarketSnapshot = components["schemas"]["MarketSnapshot"];

// Asset
export type AssetInfo = components["schemas"]["AssetInfo"];

// Discovery
export type ScreenerResult = components["schemas"]["ScreenerResult"];

export type MarketLeadersSortBy =
  | "relative_volume"
  | "rsi_14"
  | "return_1d"
  | "volume_adv_20"
  | "atr_14_pct";

export type MarketLeadersSortDir = "desc" | "asc";

export interface MarketLeadersParams {
  sortBy: MarketLeadersSortBy;
  sortDir?: MarketLeadersSortDir;
  minPrice: number;
  minAvgVolume: number;
  limit: number;
}

// Intelligence
export type RegimeType = components["schemas"]["RegimeType"];
export type RiskLevel = components["schemas"]["RiskLevel"];
export type MarketRegime = components["schemas"]["MarketRegime"];

// URL to access api
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// System API Call
export async function getSystemStatus(): Promise<SystemStatus | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/public/system/status`, {
      next: { revalidate: 60 },
    });
    if (!res.ok) throw new Error("Failed to fetch system status");
    return res.json();
  } catch (error) {
    console.error("getSystemStatus error:", error);
    return null;
  }
}

// Market API Call
export async function getHistory(
  symbol: string,
  limit: number = 1000
): Promise<HistoryDataPoint[]> {
  const url = `${API_BASE_URL}/api/v1/public/market/history/${symbol}?limit=${limit}`;
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

export async function getMarketSnapshots(
  symbols: string[]
): Promise<MarketSnapshot[]> {
  try {
    const url = `${API_BASE_URL}/api/v1/public/market/snapshots?symbols=${symbols.join(
      ","
    )}`;
    const res = await fetch(url, { next: { revalidate: 60 } });
    if (!res.ok) throw new Error("Failed to fetch snapshots");
    return res.json();
  } catch (error) {
    console.error("getMarketSnapshots error:", error);
    return [];
  }
}

// Asset API Call
export async function searchAssets(
  query: string
): Promise<components["schemas"]["AssetInfo"][]> {
  const url = `${API_BASE_URL}/api/v1/public/assets?q=${query}`;
  const res = await fetch(url);

  if (!res.ok) {
    throw new Error("Failed to search assets");
  }
  return res.json();
}

// Discovery API Call
export async function getMarketLeaders(
  params: MarketLeadersParams
): Promise<ScreenerResult[]> {
  // Create an empty URLSearchParams object
  const queryParams = new URLSearchParams();

  // Add sort_dir if present
  if (params.sortDir) {
    queryParams.append("sort_dir", params.sortDir);
  }

  // Manually append each key-value pair.
  // This ensures correct serialization.
  queryParams.append("sort_by", params.sortBy);
  queryParams.append("min_price", params.minPrice.toString());
  queryParams.append("min_avg_volume", params.minAvgVolume.toString());
  queryParams.append("limit", params.limit.toString());

  // Now the query string will be correct, e.g., "sort_by=relative_volume&min_price=5..."
  const url = `${API_BASE_URL}/api/v1/public/discovery/market-leaders?${queryParams.toString()}`;

  try {
    const res = await fetch(url, { next: { revalidate: 60 } });
    if (!res.ok) {
      // For debugging, it's helpful to see the server's response
      const errorBody = await res.text();
      console.error("API Error Body:", errorBody);
      throw new Error(`Failed to fetch market leaders. Status: ${res.status}`);
    }
    return res.json();
  } catch (error) {
    console.error("getMarketLeaders fetch error:", error);
    return [];
  }
}

// Intelligence API call
export async function getMarketRegime(): Promise<MarketRegime | null> {
  try {
    const res = await fetch(
      `${API_BASE_URL}/api/v1/public/intelligence/regime`,
      {
        cache: "no-store",
      }
    );
    if (!res.ok) {
      throw new Error("Failed to fetch market regime");
    }
    return res.json();
  } catch (error) {
    console.error("getMarketRegime error:", error);
    return null;
  }
}
