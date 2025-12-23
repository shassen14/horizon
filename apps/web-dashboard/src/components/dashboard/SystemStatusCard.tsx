import { SystemStatus, SystemHealth } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle, AlertTriangle, Wrench } from "lucide-react";
import { cn } from "@/lib/utils"; // Import cn for conditional classes

// Helper to map status to color and icon
const statusConfig: Record<
  SystemHealth,
  {
    color: string;
    borderColor: string;
    icon: React.ReactNode;
  }
> = {
  Healthy: {
    color: "bg-green-100 text-green-800",
    borderColor: "border-l-4 border-green-500",
    icon: <CheckCircle className="mr-1 h-3 w-3" />,
  },
  Stale: {
    color: "bg-yellow-100 text-yellow-800",
    borderColor: "border-l-4 border-yellow-500",
    icon: <AlertTriangle className="mr-1 h-3 w-3" />,
  },
  Degraded: {
    color: "bg-yellow-100 text-yellow-800",
    borderColor: "border-l-4 border-yellow-500",
    icon: <AlertTriangle className="mr-1 h-3 w-3" />,
  },
  //   Error: {
  //     color: "bg-red-100 text-red-800",
  //     borderColor: "border-l-4 border-red-500",
  //     icon: <XCircle className="mr-1 h-3 w-3" />,
  //   },
  Maintenance: {
    color: "bg-blue-100 text-blue-800",
    borderColor: "border-l-4 border-blue-500",
    icon: <Wrench className="mr-1 h-3 w-3" />,
  },
};

export function SystemStatusCard({ status }: { status: SystemStatus | null }) {
  if (!status) return null;

  const config = statusConfig[status.status] || statusConfig.Maintenance;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          {/* <span className={cn("mr-2 h-2 w-2 rounded-full", dotColor)}></span> */}
          System Status
          <Badge className={cn("border-none", config.color)}>
            {config.icon}
            {status.status}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="text-sm space-y-2 text-muted-foreground">
        <p>
          Active Assets:{" "}
          <span className="font-bold text-foreground">
            {status.active_assets}
          </span>
        </p>
        <p>
          Last Daily Sync:{" "}
          {status.last_daily_update
            ? new Date(status.last_daily_update).toLocaleString()
            : "N/A"}
        </p>
        <p>
          Environment:{" "}
          <span className="font-mono text-xs bg-slate-100 px-1 py-0.5 rounded text-slate-600">
            {status.environment}
          </span>
        </p>
      </CardContent>
    </Card>
  );
}
