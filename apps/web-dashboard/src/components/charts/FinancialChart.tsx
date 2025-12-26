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
import { cleanData } from "@/lib/utils";

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

    // 1. Get the browser's timezone (e.g., "America/Chicago")
    const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

    // 2. Formatters
    const crosshairFormatter = new Intl.DateTimeFormat(undefined, {
      timeZone: userTimezone,
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "numeric",
    });

    const axisTimeFormatter = new Intl.DateTimeFormat(undefined, {
      timeZone: userTimezone,
      hour: "numeric",
      minute: "numeric",
    });

    const axisDateFormatter = new Intl.DateTimeFormat(undefined, {
      timeZone: userTimezone,
      month: "short",
      day: "numeric",
      // Optional: Add year if zooming out very far
      // year: "2-digit",
    });

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

      localization: {
        // This controls the text in the floating crosshair label
        timeFormatter: (timestamp: number) => {
          return crosshairFormatter.format(new Date(timestamp * 1000));
        },
      },

      timeScale: {
        timeVisible: true,
        secondsVisible: false,

        // Prevent scrolling past the first data point (Left)
        fixLeftEdge: true,

        // Prevent scrolling past the last data point (Right)
        fixRightEdge: true,

        // Prevent zooming out so far that bars become invisible or negative width
        minBarSpacing: 0.5,
        // This controls the text on the bottom axis
        // Use tickMarkType to decide format (0=Year, 1=Month, 2=Day, 3=Time)
        tickMarkFormatter: (
          time: number,
          tickMarkType: number,
          locale: string
        ) => {
          const date = new Date(time * 1000);

          // If the library asks for Year, Month, or Day (Type < 3), show Date.
          if (tickMarkType < 3) {
            return axisDateFormatter.format(date);
          }

          // Otherwise (Type 3 or 4), show Time.
          return axisTimeFormatter.format(date);
        },
      },
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
      // 1. Clean and Set Data (Existing logic)
      const cleanOhlc = cleanData(ohlcData as CandlestickData[]);
      candlestickSeriesRef.current.setData(cleanOhlc);

      // Auto-fit content to ensure the chart looks good on load
      if (chartRef.current) {
        chartRef.current.timeScale().fitContent();
      }
    }
  }, [ohlcData]); // This triggers whenever the parent passes new data (e.g. interval change)

  useEffect(() => {
    if (volumeSeriesRef.current && ohlcData.length > 0) {
      const coloredVolumeData: HistogramData[] = ohlcData.map((d, i) => ({
        time: d.time,
        value: volumeData[i]?.value || 0,
        color:
          d.close >= d.open
            ? "rgba(38, 166, 154, 0.5)"
            : "rgba(239, 83, 80, 0.5)",
      }));

      // Clean data before setting
      const cleanVol = cleanData(coloredVolumeData);
      volumeSeriesRef.current.setData(cleanVol as HistogramData[]);
    }
  }, [ohlcData, volumeData]);

  // == 3. Dynamic Indicator Update Loop ==
  useEffect(() => {
    if (!technicalsData || technicalsData.length === 0) return;

    INDICATOR_CONTROLS.forEach((control) => {
      const isVisible = selectedIndicators[control.id];

      control.series.forEach((seriesDef) => {
        const series = indicatorsRef.current[seriesDef.key];
        if (!series) return;

        if (isVisible) {
          // 1. Map to Raw Data or Null
          const rawData = technicalsData.map((d) => {
            // STRICT TYPING: Treat features as a nested dictionary of numbers/nulls
            // This replaces 'any' with a safe, indexable type.
            const features = d.features as Record<
              string,
              Record<string, number | null | undefined>
            >;

            const [category, field] = seriesDef.apiPath.split(".");

            // Safely access the value
            const val = features[category]?.[field];

            // GUARD CLAUSE: Ensure value is a finite number
            if (typeof val !== "number" || !isFinite(val)) {
              return null;
            }

            // Construct the base point
            const time = (new Date(d.time).getTime() / 1000) as UTCTimestamp;

            // Handle Histogram specific logic (Colors)
            if (
              seriesDef.type === "Histogram" &&
              seriesDef.colorStrategy === "sign"
            ) {
              const point: HistogramData = {
                time,
                value: val,
                color: val >= 0 ? "#26a69a" : "#ef5350",
              };
              return point;
            }

            // Default Line/Histogram logic
            const point: LineData | HistogramData = {
              time,
              value: val,
            };
            return point;
          });

          // 2. Filter Nulls (Type Guard)
          const validData = rawData.filter(
            (p): p is HistogramData | LineData => p !== null
          );

          // 3. Sort & Deduplicate (Fixes assertion errors)
          const finalData = cleanData(validData);

          // 4. Set Data
          // We cast to 'any' here only because setData signature varies slightly
          // between Line/Histogram series in TS, but the data shape is compatible.
          series.setData(finalData as LineData[]);
          series.applyOptions({ visible: true });
        } else {
          series.setData([]);
          series.applyOptions({ visible: false });
        }
      });
    });
  }, [technicalsData, selectedIndicators]);

  return <div ref={chartContainerRef} className="w-full h-[600px]" />;
}
