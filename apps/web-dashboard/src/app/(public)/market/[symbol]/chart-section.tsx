// apps/web-dashboard/src/app/(public)/market/[symbol]/chart-section.tsx

"use client";

import { useState } from 'react';
import { HistoryDataPoint } from '@/lib/api';
import { FinancialChart, IndicatorSelection, OhlcData, VolumeData } from '@/components/charts/FinancialChart';
import { ChartControls } from '@/components/charts/ChartControls';

interface ChartSectionProps {
  ohlcData: OhlcData[];
  volumeData: VolumeData[];
  technicalsData: HistoryDataPoint[];
}

export function ChartSection({ ohlcData, volumeData, technicalsData }: ChartSectionProps) {
  const [selectedIndicators, setSelectedIndicators] = useState<IndicatorSelection>({
    sma50: false, 
    sma200: false,
    rsi14: false, 
  });

  return (
    <div>
      <ChartControls selection={selectedIndicators} onSelectionChange={setSelectedIndicators} />
      <FinancialChart
        ohlcData={ohlcData}
        volumeData={volumeData}
        technicalsData={technicalsData}
        selectedIndicators={selectedIndicators}
      />
    </div>
  );
}