// src/hooks/useWatchlist.ts

"use client";

import { useState, useEffect, useCallback } from "react";

const STORAGE_KEY = "horizon_watchlist";
const EVENT_KEY = "horizon_watchlist_event";

export function useWatchlist() {
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);

  // Helper to safely read from localStorage
  const getStoredWatchlist = (): string[] => {
    if (typeof window === "undefined") return [];
    try {
      const item = window.localStorage.getItem(STORAGE_KEY);
      return item ? JSON.parse(item) : [];
    } catch (error) {
      console.error("Error reading watchlist:", error);
      return [];
    }
  };

  // 1. Sync Logic (The "Real Time" Fix)
  useEffect(() => {
    // Load initial state
    setWatchlist(getStoredWatchlist());
    setIsLoaded(true);

    // Handler to update state when event fires
    const handleStorageChange = () => {
      setWatchlist(getStoredWatchlist());
    };

    // Listen for our custom event (Same Tab Sync)
    window.addEventListener(EVENT_KEY, handleStorageChange);

    // Listen for native storage event (Cross Tab/Window Sync)
    window.addEventListener("storage", (e) => {
      if (e.key === STORAGE_KEY) handleStorageChange();
    });

    return () => {
      window.removeEventListener(EVENT_KEY, handleStorageChange);
      window.removeEventListener("storage", handleStorageChange);
    };
  }, []);

  // 2. Actions
  const toggleSymbol = useCallback((symbol: string) => {
    const upper = symbol.toUpperCase();

    // Read fresh state directly from storage to avoid race conditions
    const currentList = getStoredWatchlist();

    let newList: string[];
    if (currentList.includes(upper)) {
      newList = currentList.filter((s) => s !== upper);
    } else {
      newList = [...currentList, upper];
    }

    // Save to Storage
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newList));

    // Update Local State (Immediate UI feedback for the component that clicked)
    setWatchlist(newList);

    // --- THE FIX: Dispatch Event to notify OTHER components ---
    window.dispatchEvent(new Event(EVENT_KEY));
  }, []);

  const isInWatchlist = useCallback(
    (symbol: string) => {
      return watchlist.includes(symbol.toUpperCase());
    },
    [watchlist]
  );

  return { watchlist, toggleSymbol, isInWatchlist, isLoaded };
}
