// apps/web-dashboard/src/config/indicators.ts

import { FeaturePath } from "@/types/utils";

// 1. Define the shape of a Chart Series
export interface SeriesDefinition {
  key: string; // Unique ID for the chart series (e.g., 'bb_upper')
  color: string;
  type: "Line" | "Histogram";
  apiPath: FeaturePath; // api name (e.g. trend.sma_50)
  pane: "overlay" | "separate"; // 'overlay' shares main scale, 'separate' gets its own
  lineWidth?: 1 | 2 | 3 | 4;
  lineStyle?: 0 | 1 | 2 | 3 | 4; // 0=Solid, 2=Dashed
  // 'static': Use the single color defined above (default)
  // 'sign': Green if >= 0, Red if < 0
  colorStrategy?: "static" | "sign";
}

// 2. Define the UI Control (The Button)
export interface IndicatorControl {
  id: string; // Unique ID for the toggle state (e.g., 'bb')
  label: string; // Button text
  group: "Trend" | "Momentum" | "Volatility" | "Volume"; // For UI grouping
  series: SeriesDefinition[]; // One button can toggle multiple series (e.g. Bollinger)
}

// 3. THE MASTER CONFIG
export const INDICATOR_CONTROLS: IndicatorControl[] = [
  // --- TREND ---
  {
    id: "sma20",
    label: "SMA 20",
    group: "Trend",
    series: [
      {
        key: "sma20",
        color: "#FFD700",
        type: "Line",
        apiPath: "trend.sma_20",
        pane: "overlay",
      },
    ],
  },
  {
    id: "sma50",
    label: "SMA 50",
    group: "Trend",
    series: [
      {
        key: "sma50",
        color: "#FFA500",
        type: "Line",
        apiPath: "trend.sma_50",
        pane: "overlay",
        lineWidth: 2,
      },
    ],
  },
  {
    id: "sma200",
    label: "SMA 200",
    group: "Trend",
    series: [
      {
        key: "sma200",
        color: "#800080",
        type: "Line",
        apiPath: "trend.sma_200",
        pane: "overlay",
        lineWidth: 2,
      },
    ],
  },
  {
    id: "ema12",
    label: "EMA 12",
    group: "Trend",
    series: [
      {
        key: "ema12",
        color: "#87CEEB",
        type: "Line",
        apiPath: "trend.ema_12",
        pane: "overlay",
      },
    ],
  },
  {
    id: "ema20",
    label: "EMA 20",
    group: "Trend",
    series: [
      {
        key: "ema20",
        color: "#4682B4",
        type: "Line",
        apiPath: "trend.ema_20",
        pane: "overlay",
      },
    ],
  },
  {
    id: "ema50",
    label: "EMA 50",
    group: "Trend",
    series: [
      {
        key: "ema50",
        color: "#1E90FF",
        type: "Line",
        apiPath: "trend.ema_50",
        pane: "overlay",
      },
    ],
  },
  {
    id: "macd",
    label: "MACD",
    group: "Trend",
    series: [
      {
        key: "macd_line",
        color: "#2962FF",
        type: "Line",
        apiPath: "trend.macd",
        pane: "separate",
      },
      {
        key: "macd_sig",
        color: "#FF6D00",
        type: "Line",
        apiPath: "trend.macd_signal",
        pane: "separate",
      },
      {
        key: "macd_hist",
        color: "#B2DFDB",
        type: "Histogram",
        apiPath: "trend.macd_hist",
        pane: "separate",
        colorStrategy: "sign",
      },
    ],
  },

  // --- MOMENTUM ---
  {
    id: "rsi14",
    label: "RSI 14",
    group: "Momentum",
    series: [
      {
        key: "rsi14",
        color: "#7B1FA2",
        type: "Line",
        apiPath: "momentum.rsi_14",
        pane: "separate",
      },
    ],
  },

  // --- VOLATILITY ---
  {
    id: "bb",
    label: "Bollinger Bands",
    group: "Volatility",
    series: [
      {
        key: "bb_upper",
        color: "#00897B",
        type: "Line",
        apiPath: "volatility.bb_upper_20",
        pane: "overlay",
      },
      {
        key: "bb_mid",
        color: "#4DB6AC",
        type: "Line",
        apiPath: "volatility.bb_middle_20",
        pane: "overlay",
        lineStyle: 2,
      }, // Dashed
      {
        key: "bb_lower",
        color: "#00897B",
        type: "Line",
        apiPath: "volatility.bb_lower_20",
        pane: "overlay",
      },
    ],
  },
  {
    id: "atr14",
    label: "ATR 14",
    group: "Volatility",
    series: [
      {
        key: "atr14",
        color: "#D84315",
        type: "Line",
        apiPath: "volatility.atr_14",
        pane: "separate",
      },
    ],
  },

  // --- VOLUME ---
  {
    id: "rvol",
    label: "Relative Volume",
    group: "Volume",
    series: [
      {
        key: "rvol",
        color: "#607D8B",
        type: "Histogram",
        apiPath: "volume.relative_volume",
        pane: "separate",
      },
    ],
  },
];
// --- Helpers to derive Types and State automatically ---

// Flatten the config to get a map of all series for the Chart component
export const ALL_SERIES_CONFIG = INDICATOR_CONTROLS.flatMap((c) => c.series);

// Helper to generate initial state (e.g., { sma50: true, bb: false })
export function getInitialIndicatorState(): Record<string, boolean> {
  const state: Record<string, boolean> = {};
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  INDICATOR_CONTROLS.forEach((ctrl) => {
    // Set default visibility
    // state[ctrl.id] = ctrl.id === "sma50"; // Example logic
  });
  return state;
}
