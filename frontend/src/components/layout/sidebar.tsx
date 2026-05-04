import { Link, useLocation } from "react-router-dom";
import { LayoutDashboard, Brain, Table, BarChart3, Zap, Shield, Menu, Layers } from "lucide-react";
import { cn } from "@/lib/utils";
import { DatasetSwitcher } from "./dataset-switcher";
import { Sheet, SheetTrigger, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { useState } from "react";

const NAV_ITEMS = [
  { path: "/", label: "Dashboard", icon: LayoutDashboard },
  { path: "/models", label: "Models", icon: Brain },
  { path: "/transactions", label: "Transactions", icon: Table },
  { path: "/analytics", label: "Analytics", icon: BarChart3 },
  { path: "/predict", label: "Real-time Predict", icon: Zap },
  { path: "/drift", label: "Feature Analysis", icon: Layers },
];

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const { pathname } = useLocation();

  return (
    <>
      <div className="p-6 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-emerald-500/10 flex items-center justify-center">
            <Shield className="w-5 h-5 text-emerald-500" />
          </div>
          <div>
            <h1 className="font-semibold text-sm">Fraud Detection</h1>
            <p className="text-xs text-muted-foreground">ML Dashboard</p>
          </div>
        </div>
      </div>

      <div className="p-4">
        <DatasetSwitcher />
      </div>

      <nav className="flex-1 px-3 space-y-1">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              onClick={onNavigate}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
                isActive
                  ? "bg-accent text-accent-foreground font-medium"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
              )}
            >
              <item.icon className="w-4 h-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-border">
        <p className="text-xs text-muted-foreground">
          Diploma Project — 2025
        </p>
        <p className="text-xs text-muted-foreground">
          Anomaly Detection in Financial Transactions
        </p>
      </div>
    </>
  );
}

export function Sidebar() {
  return (
    <aside className="hidden lg:flex fixed inset-y-0 left-0 z-30 w-64 border-r border-border bg-card flex-col">
      <SidebarContent />
    </aside>
  );
}

export function MobileSidebar() {
  const [open, setOpen] = useState(false);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger className="lg:hidden p-2 -ml-2 rounded-md hover:bg-accent transition-colors">
        <Menu className="w-5 h-5" />
      </SheetTrigger>
      <SheetContent side="left" className="w-64 p-0">
        <SheetTitle className="sr-only">Navigation</SheetTitle>
        <SidebarContent onNavigate={() => setOpen(false)} />
      </SheetContent>
    </Sheet>
  );
}
