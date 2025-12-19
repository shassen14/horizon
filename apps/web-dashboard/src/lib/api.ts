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

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
