import { useState } from "react";
import { GlobalSvgDefs } from "@/components/global-svg-defs";
import { TabNav } from "@/components/tab-nav";
import { ScoreOrderTab } from "@/pages/ScoreOrderTab";
import { AbuseRingTab } from "@/pages/AbuseRingTab";
import { MetricsTab } from "@/pages/MetricsTab";

type TabId = "score" | "ring" | "metrics";

function App() {
  const [tab, setTab] = useState<TabId>("score");

  return (
    <div className="relative min-h-screen p-6 md:p-10">
      <GlobalSvgDefs />
      <div className="bg-mesh">
        <div className="blob blob-violet" />
        <div className="blob blob-lime" />
        <div className="blob blob-amber" />
      </div>
      <div className="bg-grain" />

      <div className="relative z-10 mx-auto max-w-6xl">
        <header className="mb-8 flex items-center justify-between rounded-2xl bg-white/5 px-6 py-4 backdrop-blur-xl">
          <div className="flex items-center gap-3">
            <div className="glow-dot h-2.5 w-2.5 rounded-full" style={{ background: "var(--color-accent-lime)" }} />
            <span className="font-display text-lg font-semibold tracking-tight">RiskGuard</span>
            <span className="hidden font-mono text-xs text-gray-500 md:inline">/ merchant risk manager</span>
          </div>
          <TabNav
            items={[
              { id: "score", label: "Score Order" },
              { id: "ring", label: "Abuse Ring" },
              { id: "metrics", label: "Metrics" },
            ]}
            selected={tab}
            onChange={(id) => setTab(id as TabId)}
          />
        </header>

        {tab === "score" && <ScoreOrderTab />}
        {tab === "ring" && <AbuseRingTab />}
        {tab === "metrics" && <MetricsTab />}
      </div>
    </div>
  );
}

export default App;
