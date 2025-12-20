"use client";

import { useState } from "react";
import { HistoryDataPoint } from "@/lib/api";
import { OhlcData, VolumeData } from "@/types/chart";
import { getInitialIndicatorState } from "@/config/indicators";
import { FinancialChart } from "@/components/charts/FinancialChart";
import { ChartControls } from "@/components/charts/ChartControls";

interface ChartSectionProps {
  ohlcData: OhlcData[];
  volumeData: VolumeData[];
  technicalsData: HistoryDataPoint[];
}

export function ChartSection({
  ohlcData,
  volumeData,
  technicalsData,
}: ChartSectionProps) {
  // Initialize state based on the master config
  // This automatically sets defaults (e.g., SMA50=true) defined in your config
  const [selectedIndicators, setSelectedIndicators] = useState(
    getInitialIndicatorState()
  );

  return (
    <div className="space-y-4">
      {/* 
        1. Controls Component 
        Updates the state when buttons are clicked
      */}
      <ChartControls
        selection={selectedIndicators}
        onSelectionChange={setSelectedIndicators}
      />

      {/* 
        2. Chart Component
        Receives the data and the current state to render lines
      */}
      <div className="border rounded-lg p-4 bg-white shadow-sm">
        <FinancialChart
          ohlcData={ohlcData}
          volumeData={volumeData}
          technicalsData={technicalsData}
          selectedIndicators={selectedIndicators}
        />
      </div>
    </div>
  );
}
