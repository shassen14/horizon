// apps/web-dashboard/src/components/charts/FinancialChart.tsx

"use client"; 

import React, { useEffect, useRef } from 'react';
import { 
  createChart, 
  ColorType, 
  IChartApi, 
  ISeriesApi, 
  UTCTimestamp, 
  LineData,
  HistogramData,
  CandlestickData,
  CandlestickSeries,
  HistogramSeries,
  LineSeries
} from 'lightweight-charts';
import { HistoryDataPoint } from '@/lib/api';

// Define the shape of the data this component expects
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

// Define which indicators can be toggled
export type IndicatorSelection = {
  sma50: boolean;
  sma200: boolean;
  rsi14: boolean;
};

interface FinancialChartProps {
  ohlcData: OhlcData[];
  volumeData: VolumeData[];
  technicalsData: HistoryDataPoint[]; // Pass the full history for features
  selectedIndicators: IndicatorSelection;
}

export function FinancialChart({ 
  ohlcData, 
  volumeData,
  technicalsData,
  selectedIndicators
}: FinancialChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);

  // Use refs to hold stable references to the chart and series instances
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const sma50SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const sma200SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const rsiSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  // == 1. Initialization Effect == 
  // This runs only once when the component mounts.
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create the main chart instance
    const chart = createChart(chartContainerRef.current, {
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: 'black', attributionLogo: false },
      width: chartContainerRef.current.clientWidth,
      height: 450,
      grid: { vertLines: { color: '#e1e1e1' }, horzLines: { color: '#e1e1e1' } },
      rightPriceScale: { scaleMargins: { top: 0.1, bottom: 0.25 } },
      timeScale: { timeVisible: true, secondsVisible: false },
    });
    chartRef.current = chart;

    // Add all series at once. We will toggle their visibility later.
    // 1. Candlestick Series
    candlestickSeriesRef.current = chart.addSeries(CandlestickSeries, {
        upColor: '#26a69a', downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });

    // 2. Volume Series (on its own price scale)
    volumeSeriesRef.current = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume_scale',
    });
    // Configure the volume price scale to be at the bottom
    chart.priceScale('volume_scale').applyOptions({
        scaleMargins: { top: 0.7, bottom: 0 },
        entireTextOnly: true
    });
    
    // 3. Indicator Series
    sma50SeriesRef.current = chart.addSeries(LineSeries, { color: 'orange', lineWidth: 2, priceLineVisible: false, lastValueVisible: false, visible: false });
    sma200SeriesRef.current = chart.addSeries(LineSeries, { color: 'purple', lineWidth: 2, priceLineVisible: false, lastValueVisible: false, visible: false });
    
    // RSI has its own price scale
    rsiSeriesRef.current = chart.addSeries(LineSeries, {
        color: '#2962FF',
        lineWidth: 2,
        priceScaleId: 'rsi_scale',
        visible: false
    });
    chart.priceScale('rsi_scale').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
    });

    // Handle window resizing
    const handleResize = () => chart.applyOptions({ width: chartContainerRef.current?.clientWidth });
    window.addEventListener('resize', handleResize);

    // Cleanup function: runs when the component unmounts
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []); // Empty dependency array means this runs only once.

  // == 2. Data Update Effects ==
  // These run whenever the data props change.

  useEffect(() => {
    if (candlestickSeriesRef.current) {
      candlestickSeriesRef.current.setData(ohlcData as CandlestickData[]);
    }
  }, [ohlcData]);

  useEffect(() => {
    if (volumeSeriesRef.current) {
      // Color volume bars based on price change
      const coloredVolumeData = ohlcData.map((d, i) => ({
        time: d.time,
        value: volumeData[i]?.value || 0,
        color: d.close >= d.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
      }));
      volumeSeriesRef.current.setData(coloredVolumeData as HistogramData[]);
    }
  }, [ohlcData, volumeData]);

  // == 3. Indicator Toggle Effects ==
  // These run when technicals data or the selection state changes.

  useEffect(() => {
    const updateLineSeries = (
        seriesRef: React.MutableRefObject<ISeriesApi<"Line"> | null>, 
        isVisible: boolean, 
        dataKey: string
    ) => {
      const series = seriesRef.current;
      if (!series) return;
      
      if (isVisible && technicalsData) {
        const lineData = technicalsData
          .map(d => {
            // This is a common pattern for accessing nested object properties via a string path.
            const value = dataKey.split('.').reduce((obj, key) => obj?.[key], d.features as any);

            return {
              time: (new Date(d.time).getTime() / 1000) as UTCTimestamp,
              // We assert that we expect 'value' to be a number.
              value: value as number,
            };
          })
          // Filter out any points where the value is null, undefined, or not a valid number.
          .filter(d => d.value != null && isFinite(d.value))
          // IMPORTANT: The API returns data latest-first, but the chart needs it sorted oldest-first.
          .sort((a, b) => a.time - b.time); 
        
        series.setData(lineData as LineData[]);
        series.applyOptions({ visible: true });
      } else {
        series.setData([]); // Clear data when hiding
        series.applyOptions({ visible: false });
      }
    };
    updateLineSeries(sma50SeriesRef, selectedIndicators.sma50, 'trend.sma_50');
    updateLineSeries(sma200SeriesRef, selectedIndicators.sma200, 'trend.sma_200');
    updateLineSeries(rsiSeriesRef, selectedIndicators.rsi14, 'momentum.rsi_14');

  }, [technicalsData, selectedIndicators]);

  return <div ref={chartContainerRef} className="w-full h-[450px]" />;
}