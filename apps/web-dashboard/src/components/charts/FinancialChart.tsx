// apps/web-dashboard/src/components/charts/FinancialChart.tsx

"use client";

import React, { useEffect, useRef } from "react";
import {
  createChart,
  ColorType,
  IChartApi,
  ISeriesApi,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  CandlestickData,
  HistogramData,
  LineData,
  UTCTimestamp,
  LineWidth,
  LineStyle,
} from "lightweight-charts";
import { HistoryDataPoint } from "@/lib/api";
import { OhlcData, VolumeData } from "@/types/chart";
import { INDICATOR_CONTROLS, ALL_SERIES_CONFIG } from "@/config/indicators";

interface FinancialChartProps {
  ohlcData: OhlcData[];
  volumeData: VolumeData[];
  technicalsData: HistoryDataPoint[];
  // This allows the component to accept ANY keys defined in your config/indicators.ts
  selectedIndicators: Record<string, boolean>;
}

export function FinancialChart({
  ohlcData,
  volumeData,
  technicalsData,
  selectedIndicators,
}: FinancialChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  // Stable references for the main series
  const candlestickSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);

  // Dynamic dictionary to hold all indicator series instances
  // Key = series.key (from config), Value = Series Instance
  const indicatorsRef = useRef<
    Record<string, ISeriesApi<"Line" | "Histogram">>
  >({});

  // == 1. Initialization Effect ==
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create the main chart instance
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "black",
        attributionLogo: false,
      },
      width: chartContainerRef.current.clientWidth,
      height: 600, // Taller to accommodate panes
      grid: {
        vertLines: { color: "#e1e1e1" },
        horzLines: { color: "#e1e1e1" },
      },
      rightPriceScale: { scaleMargins: { top: 0.1, bottom: 0.3 } }, // Reserve bottom 30% for oscillators
      timeScale: { timeVisible: true, secondsVisible: false },
    });
    chartRef.current = chart;

    //Add Main Series (Candlesticks)
    candlestickSeriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });

    // Add Volume Series (Overlay at bottom of main pane)
    volumeSeriesRef.current = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "", // Overlay on main scale
    });
    volumeSeriesRef.current.priceScale().applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 }, // Push to very bottom
    });

    // 4. Initialize ALL Indicators from Config
    // We create every possible series defined in config, but set them to invisible.
    ALL_SERIES_CONFIG.forEach((seriesConfig) => {
      // Determine Scaling Logic
      // 'separate' -> Creates a new pane using the key as ID
      // 'overlay'  -> Locks to 'right' (main price scale)
      const priceScaleId =
        seriesConfig.pane === "separate" ? seriesConfig.key : "right";

      const commonOptions = {
        color: seriesConfig.color,
        visible: false,
        priceScaleId: priceScaleId,
        lastValueVisible: false,
        priceLineVisible: false,
        // Pass the line style and width from config, or default
        lineWidth: (seriesConfig.lineWidth || 1) as LineWidth,
        lineStyle: (seriesConfig.lineStyle || 0) as LineStyle, // 0=Solid, 1=Dotted, 2=Dashed
      };

      let series;
      if (seriesConfig.type === "Histogram") {
        series = chart.addSeries(HistogramSeries, commonOptions);
      } else {
        series = chart.addSeries(LineSeries, commonOptions);
      }

      // Configure Layout for Separate Panes
      if (seriesConfig.pane === "separate") {
        chart.priceScale(seriesConfig.key).applyOptions({
          scaleMargins: { top: 0.8, bottom: 0 }, // Squish into bottom strip
        });
      }

      // Store reference for updates later
      indicatorsRef.current[seriesConfig.key] = series;
    });

    // Handle Resizing
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, []); // Runs once on mount

  // Data Updates (OHLC & Volume)
  useEffect(() => {
    if (candlestickSeriesRef.current) {
      candlestickSeriesRef.current.setData(ohlcData as CandlestickData[]);
    }
  }, [ohlcData]);

  useEffect(() => {
    if (volumeSeriesRef.current && ohlcData.length > 0) {
      const coloredVolumeData = ohlcData.map((d, i) => ({
        time: d.time,
        value: volumeData[i]?.value || 0,
        color:
          d.close >= d.open
            ? "rgba(38, 166, 154, 0.5)"
            : "rgba(239, 83, 80, 0.5)",
      }));
      volumeSeriesRef.current.setData(coloredVolumeData as HistogramData[]);
    }
  }, [ohlcData, volumeData]);

  // == 3. Dynamic Indicator Update Loop ==
  useEffect(() => {
    if (!technicalsData || technicalsData.length === 0) return;

    // Loop through the UI CONTROLS (User Intent)
    INDICATOR_CONTROLS.forEach((control) => {
      const isVisible = selectedIndicators[control.id];

      // For each control, update ALL its associated series (e.g. BB has 3 series)
      control.series.forEach((seriesDef) => {
        const series = indicatorsRef.current[seriesDef.key];
        if (!series) return;

        if (isVisible) {
          // Map and Transform Data
          const rawData = technicalsData.map((d) => {
            // This tells TS: "It's an object where keys are strings, and values are numbers or edge cases."
            const features = d.features as Record<
              string,
              Record<string, number | null | undefined>
            >;
            const [category, field] = seriesDef.apiPath.split(".");

            const val = features[category]?.[field];

            // If val is missing or invalid, bail out early.
            // This prevents the "possibly null" error later in the code.
            if (typeof val !== "number" || !isFinite(val)) {
              return null;
            }

            // Base data point object
            const point: HistogramData | LineData = {
              time: (new Date(d.time).getTime() / 1000) as UTCTimestamp,
              value: val as number,
            };

            if (
              seriesDef.type === "Histogram" &&
              seriesDef.colorStrategy === "sign"
            ) {
              // Use standard candlestick colors: Green for positive, Red for negative
              point.color = val >= 0 ? "#26a69a" : "#ef5350";
            }

            return point;
          });

          // 2. Filter out the nulls
          // The type predicate '(p): p is ...' ensures the resulting array is clean types
          const cleanData = rawData
            .filter((p): p is HistogramData | LineData => p !== null)
            .sort((a, b) => (a.time as number) - (b.time as number));

          series.setData(cleanData);
          series.applyOptions({ visible: true });
        } else {
          series.setData([]); // Clear memory
          series.applyOptions({ visible: false });
        }
      });
    });
  }, [technicalsData, selectedIndicators]);

  return <div ref={chartContainerRef} className="w-full h-[600px]" />;
}
