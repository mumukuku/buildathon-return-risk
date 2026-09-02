import { useEffect, useMemo, useState } from "react";
import { GlassCard } from "@/components/glass-card";
import { api } from "@/lib/api";

interface DeploymentConfig {
  deployed_model: string;
  deployed_threshold: number;
  roc_auc: number;
  avg_precision: number;
  fp_review_cost_rs: number;
  max_flag_rate_assumption: number;
}
interface RingDetectorConfig {
  roc_auc: number;
  avg_precision: number;
  deployed_threshold: number;
  note: string;
}
interface BaselineRow {
  name: string;
  precision: number;
  recall: number;
  f1: number;
  flag_rate: number;
  total_cost: number;
}
interface BusinessImpactProjection {
  monthly_order_volume: number;
  projected_monthly_savings_rs: number;
  projected_annual_savings_rs: number;
}
interface BusinessImpact {
  savings_per_order_rs: number;
  savings_in_window_rs: number;
  test_window_days: number;
  projections: BusinessImpactProjection[];
  caveat: string;
}
interface CostSensitivityGrid {
  fp_costs: number[];
  max_flag_rates: number[];
  grid: {
    fp_cost: number;
    max_flag_rate: number;
    threshold: number;
    flag_rate: number;
    total_cost: number;
    precision: number;
    recall: number;
  }[];
}
interface MetricsResponse {
  deployment_config: DeploymentConfig;
  ring_detector_config: RingDetectorConfig;
  baseline_comparison: BaselineRow[];
  business_impact: BusinessImpact;
  cost_sensitivity_grid: CostSensitivityGrid;
}

function StatCard({ label, value, sublabel, glow }: { label: string; value: string; sublabel?: string; glow: string }) {
  return (
    <GlassCard glowColor={glow} className="p-5">
      <div className="field-label mb-2">{label}</div>
      <div className="font-display text-2xl font-bold">{value}</div>
      {sublabel && <div className="mt-1 text-xs text-gray-500">{sublabel}</div>}
    </GlassCard>
  );
}

const fmtRs = (n: number) => `Rs. ${Math.round(n).toLocaleString("en-IN")}`;
const fmtPct = (n: number) => `${(n * 100).toFixed(1)}%`;

export function MetricsTab() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Indices into fp_costs / max_flag_rates (defaults: fp_cost=150, max_flag_rate=0.20,
  // matching what's actually deployed). Hooks must run unconditionally, so these
  // live above the loading/error early returns below.
  const [fpCostIdx, setFpCostIdx] = useState(2);
  const [flagCapIdx, setFlagCapIdx] = useState(3);

  useEffect(() => {
    api
      .metrics()
      .then((res) => setData(res as unknown as MetricsResponse))
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load metrics. Is the local API running?"));
  }, []);

  const activeCell = useMemo(() => {
    if (!data) return null;
    const fpCost = data.cost_sensitivity_grid.fp_costs[fpCostIdx];
    const flagCap = data.cost_sensitivity_grid.max_flag_rates[flagCapIdx];
    return data.cost_sensitivity_grid.grid.find((g) => g.fp_cost === fpCost && g.max_flag_rate === flagCap) ?? null;
  }, [data, fpCostIdx, flagCapIdx]);

  if (error) {
    return <p className="text-sm text-[var(--color-tier-decline)]">{error}</p>;
  }
  if (!data) {
    return <p className="text-sm text-gray-500">Loading metrics…</p>;
  }

  const { deployment_config: dc, ring_detector_config: rc, baseline_comparison: baseline, business_impact: bi } = data;
  const ml = baseline.find((r) => r.name.includes("XGBoost")) ?? baseline[1];
  const rule = baseline.find((r) => r.name.includes("Rule")) ?? baseline[0];
  const savingsVsBaseline = rule.total_cost - ml.total_cost;
  const { fp_costs: fpCosts, max_flag_rates: flagCaps } = data.cost_sensitivity_grid;

  return (
    <div className="space-y-6">
      {/* Headline stats */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard label="Return-risk ROC-AUC" value={dc.roc_auc.toFixed(3)} glow="var(--color-accent-violet)" />
        <StatCard label="Deployed threshold" value={dc.deployed_threshold.toFixed(2)} sublabel={`max ${fmtPct(dc.max_flag_rate_assumption)} flag rate`} glow="var(--color-tier-review)" />
        <StatCard label="Ring detector ROC-AUC" value={rc.roc_auc.toFixed(3)} sublabel="small sample -- see caveats" glow="var(--color-accent-lime)" />
        <StatCard label="Savings vs. rule baseline" value={fmtRs(savingsVsBaseline)} sublabel="on held-out test set" glow="var(--color-tier-approve)" />
      </div>

      {/* Interactive cost-sensitivity slider */}
      <GlassCard glowColor="var(--color-tier-review)" className="p-6">
        <h3 className="font-display text-lg font-semibold">Cost-sensitivity explorer</h3>
        <p className="mb-6 text-sm text-gray-500">
          Drag either slider to see how the assumed false-positive review cost and the fraud-ops team's review
          capacity change the cost-optimal threshold. Every point here is a real re-run of the same optimization
          used to pick our actual deployed threshold -- not interpolated or simulated.
        </p>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <div>
            <div className="mb-2 flex justify-between">
              <label className="field-label" htmlFor="fp-cost-slider">
                False-positive review cost
              </label>
              <span className="font-mono text-xs text-[var(--color-tier-review)]">{fmtRs(fpCosts[fpCostIdx])}</span>
            </div>
            <input
              id="fp-cost-slider"
              type="range"
              min={0}
              max={fpCosts.length - 1}
              step={1}
              value={fpCostIdx}
              onChange={(e) => setFpCostIdx(Number(e.target.value))}
              className="w-full accent-[var(--color-accent-lime)]"
            />
          </div>
          <div>
            <div className="mb-2 flex justify-between">
              <label className="field-label" htmlFor="flag-cap-slider">
                Max review capacity (flag rate)
              </label>
              <span className="font-mono text-xs text-[var(--color-accent-lime)]">{fmtPct(flagCaps[flagCapIdx])}</span>
            </div>
            <input
              id="flag-cap-slider"
              type="range"
              min={0}
              max={flagCaps.length - 1}
              step={1}
              value={flagCapIdx}
              onChange={(e) => setFlagCapIdx(Number(e.target.value))}
              className="w-full accent-[var(--color-accent-lime)]"
            />
          </div>
        </div>

        {activeCell && (
          <div className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="rounded-xl bg-white/5 p-4">
              <div className="font-mono text-xs text-gray-500">Resulting threshold</div>
              <div className="mt-1 font-display text-xl font-bold">{activeCell.threshold.toFixed(2)}</div>
            </div>
            <div className="rounded-xl bg-white/5 p-4">
              <div className="font-mono text-xs text-gray-500">Actual flag rate</div>
              <div className="mt-1 font-display text-xl font-bold">{fmtPct(activeCell.flag_rate)}</div>
            </div>
            <div className="rounded-xl bg-white/5 p-4">
              <div className="font-mono text-xs text-gray-500">Total cost (test set)</div>
              <div className="mt-1 font-display text-xl font-bold">{fmtRs(activeCell.total_cost)}</div>
            </div>
            <div className="rounded-xl bg-white/5 p-4">
              <div className="font-mono text-xs text-gray-500">Precision / Recall</div>
              <div className="mt-1 font-display text-xl font-bold">
                {activeCell.precision.toFixed(2)} / {activeCell.recall.toFixed(2)}
              </div>
            </div>
          </div>
        )}
      </GlassCard>

      {/* Baseline comparison */}
      <GlassCard glowColor="var(--color-accent-violet)" className="p-6">
        <h3 className="font-display text-lg font-semibold">ML vs. rule-based baseline</h3>
        <p className="mb-4 text-sm text-gray-500">
          A fixed-threshold heuristic a merchant might use without ML, compared to our deployed model on the same
          held-out test set.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-gray-500">
                <th className="pb-2 pr-4">Model</th>
                <th className="pb-2 pr-4">Precision</th>
                <th className="pb-2 pr-4">Recall</th>
                <th className="pb-2 pr-4">F1</th>
                <th className="pb-2 pr-4">Flag rate</th>
                <th className="pb-2">Total cost</th>
              </tr>
            </thead>
            <tbody className="font-mono text-xs">
              {baseline.map((row) => (
                <tr key={row.name} className="border-b border-white/5">
                  <td className="py-2 pr-4 font-sans text-sm">{row.name}</td>
                  <td className="py-2 pr-4">{row.precision.toFixed(3)}</td>
                  <td className="py-2 pr-4">{row.recall.toFixed(3)}</td>
                  <td className="py-2 pr-4">{row.f1.toFixed(3)}</td>
                  <td className="py-2 pr-4">{fmtPct(row.flag_rate)}</td>
                  <td className="py-2">{fmtRs(row.total_cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </GlassCard>

      {/* Business impact */}
      <GlassCard glowColor="var(--color-tier-approve)" className="p-6">
        <h3 className="font-display text-lg font-semibold">Projected business impact</h3>
        <p className="mb-4 text-sm text-gray-500">
          {fmtRs(bi.savings_per_order_rs)} saved per order placed, linearly extrapolated to a few example merchant
          scales. {bi.caveat}
        </p>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {bi.projections.map((p) => (
            <div key={p.monthly_order_volume} className="rounded-xl bg-white/5 p-4">
              <div className="font-mono text-xs text-gray-500">
                {p.monthly_order_volume.toLocaleString("en-IN")} orders/mo
              </div>
              <div className="mt-1 font-display text-lg font-bold">{fmtRs(p.projected_monthly_savings_rs)}/mo</div>
              <div className="text-xs text-gray-500">{fmtRs(p.projected_annual_savings_rs)}/yr</div>
            </div>
          ))}
        </div>
      </GlassCard>

      <p className="text-center text-xs text-gray-600">{rc.note}</p>
    </div>
  );
}
