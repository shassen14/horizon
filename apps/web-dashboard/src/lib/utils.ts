import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { HistogramData, LineData } from "lightweight-charts";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Generic helper to Sort and Deduplicate chart data
export function cleanData<T extends LineData | HistogramData>(data: T[]): T[] {
  // 1. Sort ascending by time
  const sorted = [...data].sort(
    (a, b) => (a.time as number) - (b.time as number)
  );

  // 2. Deduplicate timestamps (keep the first occurrence)
  const unique: T[] = [];
  const seenTimestamps = new Set<number>();

  for (const item of sorted) {
    const timeNum = item.time as number;
    if (!seenTimestamps.has(timeNum)) {
      seenTimestamps.add(timeNum);
      unique.push(item);
    }
  }

  return unique;
}
