// apps/web-dashboard/src/components/layout/Navbar.tsx

import Link from "next/link";
import { CommandPalette } from "@/components/search/CommandPalette";
import { Activity } from "lucide-react";

export function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-14 items-center">
        {/* Logo / Brand */}
        <div className="mr-4 hidden md:flex">
          <Link href="/" className="mr-6 flex items-center space-x-2">
            <Activity className="h-6 w-6" /> {/* Example Icon */}
            <span className="hidden font-bold sm:inline-block">Horizon</span>
          </Link>
          <nav className="flex items-center gap-6 text-sm">
            <Link
              href="/"
              className="transition-colors hover:text-foreground/80 text-foreground"
            >
              Dashboard
            </Link>
            <Link
              href="/market"
              className="transition-colors hover:text-foreground/80 text-foreground/60"
            >
              Market
            </Link>
            {/* <Link
              href="/about"
              className="transition-colors hover:text-foreground/80 text-foreground/60"
            >
              About
            </Link> */}
          </nav>
        </div>

        {/* Right Side: Search & Auth (future) */}
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <div className="w-full flex-1 md:w-auto md:flex-none">
            <CommandPalette />
          </div>
        </div>
      </div>
    </header>
  );
}
