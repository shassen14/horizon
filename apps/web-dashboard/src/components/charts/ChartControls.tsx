// apps/web-dashboard/src/components/charts/ChartControls.tsx

"use client";

import { IndicatorSelection } from './FinancialChart';
import { Button } from '@/components/ui/button';

interface ChartControlsProps {
  selection: IndicatorSelection;
  onSelectionChange: (newSelection: IndicatorSelection) => void;
}

export function ChartControls({ selection, onSelectionChange }: ChartControlsProps) {
  const toggle = (key: keyof IndicatorSelection) => {
    onSelectionChange({ ...selection, [key]: !selection[key] });
  };

  return (
    <div className="flex items-center space-x-2 my-4">
      <Button variant={selection.sma50 ? 'default' : 'outline'} onClick={() => toggle('sma50')}>SMA 50</Button>
      <Button variant={selection.sma200 ? 'default' : 'outline'} onClick={() => toggle('sma200')}>SMA 200</Button>
      <Button variant={selection.rsi14 ? 'default' : 'outline'} onClick={() => toggle('rsi14')}>RSI 14</Button>
    </div>
  );
}