// apps/web-dashboard/src/types/chart.ts

import { UTCTimestamp } from "lightweight-charts";

// Standardized Data Points
export interface OhlcData {
  time: UTCTimestamp;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface VolumeData {
  time: UTCTimestamp;
  value: number;
}

export interface SingleValueData {
  time: UTCTimestamp;
  value: number;
}
