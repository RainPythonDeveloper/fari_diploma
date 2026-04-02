import { Routes, Route } from "react-router-dom";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import DashboardPage from "@/pages/Dashboard";
import ModelsPage from "@/pages/Models";
import TransactionsPage from "@/pages/Transactions";
import AnalyticsPage from "@/pages/Analytics";
import PredictPage from "@/pages/Predict";

export default function App() {
  return (
    <>
      <Sidebar />
      <div className="flex-1 flex flex-col min-h-screen lg:ml-64">
        <Header />
        <main className="flex-1 p-4 lg:p-6">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/transactions" element={<TransactionsPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/predict" element={<PredictPage />} />
          </Routes>
        </main>
      </div>
    </>
  );
}
