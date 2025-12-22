// apps/web-dashboard/src/hooks/useAssetSearch.tsx

"use client";

import { useQuery } from "@tanstack/react-query";
import { useState, useEffect } from "react";

// Define a simple type for the search result based on your API schema
interface AssetResult {
  symbol: string;
  name: string;
  exchange: string;
}

export function useAssetSearch() {
  const [query, setQuery] = useState("");
  // Debounce logic: wait 300ms after typing stops before fetching
  const [debouncedQuery, setDebouncedQuery] = useState(query);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedQuery(query);
    }, 300);
    return () => clearTimeout(handler);
  }, [query]);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const { data, isLoading } = useQuery({
    queryKey: ["assetSearch", debouncedQuery],
    queryFn: async () => {
      if (!debouncedQuery) return [];
      const res = await fetch(
        `${API_BASE}/api/v1/public/assets?q=${debouncedQuery}&limit=10`
      );
      if (!res.ok) throw new Error("Search failed");
      return res.json() as Promise<AssetResult[]>;
    },
    enabled: debouncedQuery.length > 0, // Only fetch if user typed something
    staleTime: 1000 * 60 * 5, // Cache results for 5 minutes
  });

  return { query, setQuery, data, isLoading };
}
