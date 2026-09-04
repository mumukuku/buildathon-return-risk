import { useEffect, useState } from "react";
import { GlassCard } from "@/components/glass-card";
import { Gauge } from "@/components/gauge";
import { FactorList } from "@/components/factor-bar";
import { api, type ClusterInput, type RingResult } from "@/lib/api";

function clusterSummary(c: ClusterInput): string {
  const shares: string[] = [];
  if (c.shares_device) shares.push("device");
  if (c.shares_payment) shares.push("payment");
  if (c.shares_address) shares.push("address");
  const shareText = shares.length ? `shares ${shares.join(" + ")}` : "shares nothing";
  return `${c.size} accounts, ${shareText}`;
}

export function AbuseRingTab() {
  const [clusters, setClusters] = useState<ClusterInput[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [result, setResult] = useState<RingResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    api
      .sampleClusters()
      .then((res) => setClusters(res.clusters))
      .catch((e) => setLoadError(e instanceof Error ? e.message : "Failed to load sample clusters."));
  }, []);

  async function handleCheck(idx: number) {
    setSelectedIdx(idx);
    setLoading(true);
    setError(null);
    try {
      const res = await api.checkCluster(clusters[idx]);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to check cluster. Is the local API running?");
    } finally {
      setLoading(false);
    }
  }

  const verdictColor =
    result?.verdict === "likely_ring" ? "var(--color-tier-decline)" : "var(--color-tier-approve)";

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
      <GlassCard glowColor="var(--color-accent-violet)" className="p-6 lg:col-span-3">
        <h2 className="font-display text-lg font-semibold">Check a cluster</h2>
        <p className="mb-6 text-sm text-gray-500">
          Sample clusters from training data -- some are real coordinated abuse rings, some are benign identifier
          sharing (e.g. family at one address). The ground-truth label isn't shown until after you check it.
        </p>

        {loadError && <p className="text-sm text-[var(--color-tier-decline)]">{loadError}</p>}

        <div className="space-y-2">
          {clusters.map((c, idx) => (
            <button
              key={c.cluster_id ?? idx}
              type="button"
              onClick={() => handleCheck(idx)}
              className="field-input flex w-full items-center justify-between text-left transition-colors"
              style={{
                borderColor: selectedIdx === idx ? "var(--color-accent-lime)" : undefined,
              }}
            >
              <span>{clusterSummary(c)}</span>
              <span className="font-mono text-xs text-gray-500">
                avg return rate {(c.avg_return_rate * 100).toFixed(0)}%
              </span>
            </button>
          ))}
        </div>
        {error && <p className="mt-3 text-sm text-[var(--color-tier-decline)]">{error}</p>}
      </GlassCard>

      <GlassCard glowColor={verdictColor} className="flex flex-col items-center p-6 lg:col-span-2">
        {loading ? (
          <div className="flex h-full items-center justify-center text-sm text-gray-500">Checking…</div>
        ) : result ? (
          <>
            <span className="field-label mb-4">Ring score</span>
            <Gauge value={Math.round(result.ring_score * 100)} />
            <span
              className="verdict-badge mt-3"
              style={{ color: verdictColor, background: `color-mix(in srgb, ${verdictColor} 15%, transparent)` }}
            >
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: verdictColor }} />
              {result.verdict === "likely_ring" ? "LIKELY RING" : "LIKELY BENIGN"}
            </span>
            <p className="mt-4 text-center text-sm text-gray-300">{result.plain_english_summary}</p>
            <div className="mt-6 w-full">
              <div className="field-label mb-3">Top factors</div>
              <FactorList factors={result.top_factors} />
            </div>
          </>
        ) : (
          <div className="flex h-full flex-col items-center justify-center text-center text-sm text-gray-500">
            <p>Select a cluster on the left to check whether it looks like a coordinated abuse ring.</p>
          </div>
        )}
      </GlassCard>
      </div>

      {selectedIdx !== null && clusters[selectedIdx]?.members && clusters[selectedIdx].members!.length > 0 && (
        <GlassCard glowColor={verdictColor} className="p-6">
          <h3 className="font-display text-lg font-semibold">
            Who's in this cluster ({clusters[selectedIdx].members!.length} accounts)
          </h3>
          <p className="mb-4 text-sm text-gray-500">
            The actual accounts behind the aggregate stats above -- not just "size=5, avg return rate=40%," but which
            customer IDs, how old each account is, and each member's individual return history.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-gray-500">
                  <th className="pb-2 pr-4">Customer ID</th>
                  <th className="pb-2 pr-4">Account age (days)</th>
                  <th className="pb-2 pr-4">Total orders</th>
                  <th className="pb-2 pr-4">Return rate</th>
                  <th className="pb-2">Abusive-return rate</th>
                </tr>
              </thead>
              <tbody className="font-mono text-xs">
                {clusters[selectedIdx].members!.map((m) => (
                  <tr key={m.customer_id} className="border-b border-white/5">
                    <td className="py-2 pr-4">{m.customer_id}</td>
                    <td className="py-2 pr-4">{m.account_age_days}</td>
                    <td className="py-2 pr-4">{m.total_orders}</td>
                    <td className="py-2 pr-4">{(m.return_rate * 100).toFixed(0)}%</td>
                    <td className="py-2">{(m.abusive_return_rate * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </GlassCard>
      )}
    </div>
  );
}
