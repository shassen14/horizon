import { MarketScreener } from "@/components/market/MarketScreener";

export default function MarketPage() {
  return (
    <div className="container mx-auto p-4 space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Market Discovery</h1>
        <p className="text-muted-foreground">
          Filter and sort assets to find opportunities matching your criteria.
        </p>
      </div>
      <MarketScreener />
    </div>
  );
}
