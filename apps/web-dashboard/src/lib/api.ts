// apps/web-dashboard/src/lib/api.ts

import { components } from "./api-client"; // Import the auto-generated types

// System
// Enum
export type SystemHealth = components["schemas"]["SystemHealth"];
export type Environment = components["schemas"]["Environment"];
// Return Type
export type SystemStatus = components["schemas"]["SystemStatus"];

// Market
// Enum
export type MarketInterval = components["schemas"]["MarketInterval"];
// Return Type
export type HistoryDataPoint = components["schemas"]["HistoryDataPoint"];
export type FeatureSet = components["schemas"]["FeatureSet"];
export type TrendFeatures = components["schemas"]["TrendFeatures"];
export type MomentumFeatures = components["schemas"]["MomentumFeatures"];
export type VolatilityFeatures = components["schemas"]["VolatilityFeatures"];
export type VolumeFeatures = components["schemas"]["VolumeFeatures"];
export type MarketSnapshot = components["schemas"]["MarketSnapshot"];

// Asset
export type AssetInfo = components["schemas"]["AssetInfo"];
export type AssetDetail = components["schemas"]["AssetDetail"];

// Discovery
// Enum
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
// Return Type
export type ScreenerResult = components["schemas"]["ScreenerResult"];

// Intelligence
// Enum
export type RegimeType = components["schemas"]["RegimeType"];
export type RiskLevel = components["schemas"]["RiskLevel"];
// Return Type
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
  limit: number = 1000,
  interval: MarketInterval = "1d",
  startDate?: Date,
  endDate?: Date
): Promise<HistoryDataPoint[]> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    interval: interval,
  });

  if (startDate) params.append("start_date", startDate.toISOString());
  if (endDate) params.append("end_date", endDate.toISOString());

  const url = `${API_BASE_URL}/api/v1/public/market/history/${symbol}?${params.toString()}`;

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

export async function getAssetDetail(
  symbol: string
): Promise<AssetDetail | null> {
  const url = `${API_BASE_URL}/api/v1/public/assets/${symbol}`;
  try {
    const res = await fetch(url, { next: { revalidate: 3600 } }); // Cache for 1 hour
    if (!res.ok) return null;
    return res.json();
  } catch (error) {
    console.error("getAssetDetail error:", error);
    return null;
  }
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
  // 1. Define URL explicitly for logging
  const url = `${process.env.NEXT_PUBLIC_API_URL}/public/intelligence/regime`;

  try {
    const res = await fetch(url, {
      cache: "no-store",
      headers: {
        "User-Agent": "Horizon-Dashboard/1.0",
      },
    });

    // 2. Check for HTTP Errors FIRST
    if (!res.ok) {
      const errorBody = await res.text();

      // Log critical details to the Vercel Server Logs
      console.error(`[API Error] Status: ${res.status}`);
      console.error(`[API Error] URL: ${url}`);
      console.error(`[API Error] Body Snippet: ${errorBody.substring(0, 500)}`); // First 500 chars

      throw new Error(`API responded with status ${res.status}`);
    }

    // 3. Parse JSON safely
    return await res.json();
  } catch (error) {
    // This catches network errors (DNS, Timeout) AND the error thrown above
    console.error("[getMarketRegime] Final Error:", error);
    return null;
  }
}
