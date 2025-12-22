// apps/web-dashboard/src/components/search/CommandPalette.tsx

"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { Search } from "lucide-react";
import { useAssetSearch } from "@/hooks/useAssetSearch";
import { Button } from "@/components/ui/button";

// Import primitives to build a custom dialog that respects 'shouldFilter'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { VisuallyHidden } from "@radix-ui/react-visually-hidden";

export function CommandPalette() {
  const [open, setOpen] = React.useState(false);
  const router = useRouter();

  const { query, setQuery, data: results, isLoading } = useAssetSearch();

  React.useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((open) => !open);
      }
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  const handleSelect = (symbol: string) => {
    setOpen(false);
    router.push(`/market/${symbol}`);
  };

  return (
    <>
      <Button
        variant="outline"
        className="relative h-9 w-full justify-start rounded-[0.5rem] bg-background text-sm font-normal text-muted-foreground shadow-none sm:pr-12 md:w-40 lg:w-64"
        onClick={() => setOpen(true)}
      >
        <Search className="mr-2 h-4 w-4" />
        <span className="hidden lg:inline-flex">Search market...</span>
        <span className="inline-flex lg:hidden">Search...</span>
        <kbd className="pointer-events-none absolute right-[0.3rem] top-[0.3rem] hidden h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100 sm:flex">
          <span className="text-xs">⌘</span>K
        </kbd>
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="overflow-hidden p-0 shadow-lg sm:max-w-[500px]">
          {/* Accessibility requirement for Dialog */}
          <VisuallyHidden>
            <DialogTitle>Search Market</DialogTitle>
          </VisuallyHidden>

          {/* 
            CRITICAL FIX: 
            We use the Command primitive directly here so we can pass 
            shouldFilter={false}. This stops cmdk from hiding our async results.
          */}
          <Command
            shouldFilter={false}
            className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:font-medium [&_[cmdk-group-heading]]:text-muted-foreground [&_[cmdk-item]]:px-2 [&_[cmdk-item]]:py-3 [&_[cmdk-item]_svg]:h-5 [&_[cmdk-item]_svg]:w-5"
          >
            <CommandInput
              placeholder="Type a symbol (e.g. AAPL)..."
              value={query}
              onValueChange={setQuery}
            />
            <CommandList>
              {isLoading && (
                <div className="py-6 text-center text-sm text-muted-foreground">
                  Searching...
                </div>
              )}

              {!isLoading && results?.length === 0 && query.length > 0 && (
                <CommandEmpty>No results found.</CommandEmpty>
              )}

              {!isLoading && results && results.length > 0 && (
                <CommandGroup heading="Stocks">
                  {results.map((asset) => (
                    <CommandItem
                      key={asset.symbol}
                      value={asset.symbol} // This needs to be present for selection to work
                      onSelect={() => handleSelect(asset.symbol)}
                    >
                      <Search className="mr-2 h-4 w-4 opacity-50" />
                      <span className="font-bold mr-2">{asset.symbol}</span>
                      <span className="text-muted-foreground text-xs truncate">
                        {asset.name}
                      </span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}
            </CommandList>
          </Command>
        </DialogContent>
      </Dialog>
    </>
  );
}
