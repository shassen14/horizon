import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { Time } from "lightweight-charts";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// T must just be an object with a 'time' property of type UTCTimestamp.
// It doesn't care if it has 'value', 'open', 'color', etc.
export function cleanData<T extends { time: Time }>(data: T[]): T[] {
  // 1. Sort by time ASC
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
