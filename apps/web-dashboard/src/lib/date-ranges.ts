// apps/web-dashboard/src/lib/date-ranges.ts

export type TimeRange = "1D" | "1W" | "1M" | "3M" | "YTD" | "1Y" | "5Y" | "ALL";

export function getDateRange(range: TimeRange): {
  startDate?: Date;
  endDate?: Date;
} {
  const end = new Date(); // Now
  const start = new Date(); // We will modify this

  switch (range) {
    case "1D":
      // For 1D, we usually want the last 24 hours or the current trading session.
      // Subtracting 1 day is safe.
      start.setDate(end.getDate() - 1);
      break;
    case "1W":
      start.setDate(end.getDate() - 7);
      break;
    case "1M":
      start.setMonth(end.getMonth() - 1);
      break;
    case "3M":
      start.setMonth(end.getMonth() - 3);
      break;
    case "YTD":
      start.setMonth(0); // January
      start.setDate(1); // 1st
      start.setFullYear(end.getFullYear()); // Current Year
      break;
    case "1Y":
      start.setFullYear(end.getFullYear() - 1);
      break;
    case "5Y":
      start.setFullYear(end.getFullYear() - 5);
      break;
    case "ALL":
      // Return undefined to let the API fetch everything it has
      return { startDate: undefined, endDate: undefined };
  }

  return { startDate: start, endDate: end };
}
