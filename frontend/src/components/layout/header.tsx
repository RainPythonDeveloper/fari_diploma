import { useLocation } from "react-router-dom";
import { MobileSidebar } from "./sidebar";

const PAGE_TITLES: Record<string, string> = {
  "/": "Dashboard",
  "/models": "Model Comparison",
  "/transactions": "Transaction Explorer",
  "/analytics": "Analytics",
  "/predict": "Real-time Prediction",
};

export function Header() {
  const { pathname } = useLocation();
  const title = PAGE_TITLES[pathname] || "Dashboard";

  return (
    <header className="h-14 border-b border-border flex items-center gap-3 px-4 lg:px-6 bg-card/50 backdrop-blur-sm">
      <MobileSidebar />
      <h2 className="text-lg font-semibold">{title}</h2>
    </header>
  );
}
