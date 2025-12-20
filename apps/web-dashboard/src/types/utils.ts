// apps/web-dashboard/src/types/utils.ts

import { FeatureSet } from "@/lib/api";

/**
 * This mapped type iterates over the keys of FeatureSet (trend, momentum, etc.),
 * and then iterates over the keys of those sub-objects (sma_50, rsi_14, etc.).
 *
 * Result: "trend.sma_50" | "trend.sma_200" | "momentum.rsi_14" ...
 */
export type FeaturePath = {
  [K in keyof FeatureSet]: K extends string
    ? `${K}.${keyof NonNullable<FeatureSet[K]> & string}`
    : never;
}[keyof FeatureSet];
