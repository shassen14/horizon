// apps/web-dashboard/src/components/dashboard/SparklineChart.tsx

"use client";

import { LineChart, Line, ResponsiveContainer, YAxis } from "recharts";

interface SparklineChartProps {
  data: { name: number; value: number }[];
  color: string;
}

export function SparklineChart({ data, color }: SparklineChartProps) {
  // This component will only be rendered on the client,
  // where its parent div is guaranteed to have dimensions.
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data}>
        <YAxis domain={["dataMin", "dataMax"]} hide />
        <Line
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
