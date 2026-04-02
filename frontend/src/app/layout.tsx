import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { DatasetProvider } from "@/hooks/use-dataset";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Fraud Detection Dashboard",
  description: "ML-powered anomaly detection in financial transactions",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} dark h-full antialiased`}
    >
      <body className="min-h-full flex bg-background text-foreground">
        <DatasetProvider>
          <Sidebar />
          <div className="flex-1 flex flex-col min-h-screen lg:ml-64">
            <Header />
            <main className="flex-1 p-4 lg:p-6">{children}</main>
          </div>
        </DatasetProvider>
      </body>
    </html>
  );
}
