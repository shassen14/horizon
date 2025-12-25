import { AssetDetail } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function KeyStatistics({ asset }: { asset: AssetDetail }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {/* Exchange */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Exchange
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Handle nullable exchange */}
          <div className="text-2xl font-bold">{asset.exchange || "N/A"}</div>
        </CardContent>
      </Card>

      {/* Sector / Asset Class */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Asset Class
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold capitalize">
            {/* Clean up the string (e.g., 'us_equity' -> 'Us Equity') */}
            {asset.asset_class.replace(/_/g, " ")}
          </div>
        </CardContent>
      </Card>

      {/* Status (Fixed to use is_active boolean) */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Trading Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center h-8">
            {asset.is_active ? (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-sm font-medium bg-green-100 text-green-800">
                Active
              </span>
            ) : (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-sm font-medium bg-red-100 text-red-800">
                Inactive
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Data Source (Static for now) */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Data Source
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">Alpaca</div>
        </CardContent>
      </Card>
    </div>
  );
}
