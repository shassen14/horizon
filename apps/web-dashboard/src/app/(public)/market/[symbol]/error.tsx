// apps/web-dashboard/src/app/(public)/market/[symbol]/error.tsx

"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { AlertCircle } from "lucide-react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Market Page Error:", error);
  }, [error]);

  return (
    <div className="h-[50vh] flex flex-col items-center justify-center space-y-4">
      <div className="flex items-center space-x-2 text-destructive">
        <AlertCircle className="h-6 w-6" />
        <h2 className="text-xl font-semibold">Unable to load market data</h2>
      </div>
      <p className="text-muted-foreground">
        The backend might be offline or the symbol doesn&apos;t exist in our
        database.
      </p>
      <Button variant="outline" onClick={() => reset()}>
        Try Again
      </Button>
    </div>
  );
}
