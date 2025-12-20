// apps/web-dashboard/src/components/charts/ChartControls.tsx

"use client";

import { INDICATOR_CONTROLS } from "@/config/indicators";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import {
  ChevronDown,
  Activity,
  TrendingUp,
  BarChart3,
  Waves,
} from "lucide-react";

interface ChartControlsProps {
  selection: Record<string, boolean>;
  onSelectionChange: (newSelection: Record<string, boolean>) => void;
}

export function ChartControls({
  selection,
  onSelectionChange,
}: ChartControlsProps) {
  const toggle = (id: string) => {
    onSelectionChange({ ...selection, [id]: !selection[id] });
  };

  // Groups to render
  const groups = [
    { id: "Trend", icon: TrendingUp, label: "Trend" },
    { id: "Momentum", icon: Activity, label: "Momentum" },
    { id: "Volatility", icon: Waves, label: "Volatility" },
    { id: "Volume", icon: BarChart3, label: "Volume" },
  ];

  return (
    <div className="flex flex-wrap gap-2 my-4 p-2 bg-slate-50 border rounded-lg">
      <span className="text-sm font-medium text-slate-500 flex items-center mr-2">
        Indicators:
      </span>

      {groups.map((group) => {
        const groupControls = INDICATOR_CONTROLS.filter(
          (c) => c.group === group.id
        );
        if (groupControls.length === 0) return null;

        // Count active indicators in this group for a badge effect (optional polish)
        const activeCount = groupControls.filter((c) => selection[c.id]).length;

        return (
          <DropdownMenu key={group.id}>
            <DropdownMenuTrigger asChild>
              <Button
                variant={activeCount > 0 ? "secondary" : "ghost"}
                size="sm"
                className="h-8 border-dashed border border-transparent hover:border-slate-300"
              >
                <group.icon className="mr-2 h-4 w-4" />
                {group.label}
                {activeCount > 0 && (
                  <span className="ml-2 rounded-full bg-slate-200 px-1.5 py-0.5 text-xs font-semibold text-slate-700">
                    {activeCount}
                  </span>
                )}
                <ChevronDown className="ml-2 h-3 w-3 opacity-50" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-56">
              <DropdownMenuLabel>{group.label} Indicators</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {groupControls.map((ctrl) => (
                <DropdownMenuCheckboxItem
                  key={ctrl.id}
                  checked={selection[ctrl.id]}
                  onCheckedChange={() => toggle(ctrl.id)}
                >
                  {ctrl.label}
                </DropdownMenuCheckboxItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        );
      })}
    </div>
  );
}
