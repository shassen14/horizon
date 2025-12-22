// apps/web-dashboard/src/app/(public)/market/page.tsx

import { MarketLeadersTable } from "./market-leaders-table";

export default function MarketPage() {
  return (
    <div className="container mx-auto p-4 space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Market Discovery</h1>
        <p className="text-muted-foreground">
          Find stocks with unusual activity based on daily metrics.
        </p>
      </div>

      <MarketLeadersTable />
    </div>
  );
}
