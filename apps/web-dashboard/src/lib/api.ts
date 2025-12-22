// apps/web-dashboard/src/lib/api.ts

import { components } from "./api-client"; // Import the auto-generated types

// Market
export type HistoryDataPoint = components["schemas"]["HistoryDataPoint"];
export type FeatureSet = components["schemas"]["FeatureSet"];
export type TrendFeatures = components["schemas"]["TrendFeatures"];
export type MomentumFeatures = components["schemas"]["MomentumFeatures"];
export type VolatilityFeatures = components["schemas"]["VolatilityFeatures"];
export type VolumeFeatures = components["schemas"]["VolumeFeatures"];

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

export interface MarketLeadersParams {
  sortBy: MarketLeadersSortBy;
  minPrice: number;
  minAvgVolume: number;
  limit: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Market API Endpoints
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

// Asset API Endpoints
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

// Discovery API Endpoints
export async function getMarketLeaders(
  params: MarketLeadersParams
): Promise<ScreenerResult[]> {
  // Create an empty URLSearchParams object
  const queryParams = new URLSearchParams();

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
