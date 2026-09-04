import { useState } from "react";
import { GlassCard } from "@/components/glass-card";
import { Gauge, tierFromScore } from "@/components/gauge";
import { FactorList } from "@/components/factor-bar";
import { api, type OrderInput, type ScoreResult } from "@/lib/api";

const CATEGORIES = ["apparel", "footwear", "electronics", "mobile", "beauty", "home", "accessories"];
const PAYMENT_METHODS = ["COD", "UPI", "card", "wallet"];
const RETURN_REASONS = [
  "changed_mind",
  "size_issue",
  "damaged",
  "not_as_described",
  "wrong_item",
  "no_longer_needed",
];

const DEFAULT_ORDER: OrderInput = {
  order_value: 4500,
  category: "apparel",
  payment_method: "COD",
  return_reason: "changed_mind",
  delivery_days: 3,
  order_hour: 14,
  is_weekend: 0,
  account_age_days_at_order: 200,
  hist_orders_before: 5,
  hist_return_rate_before: 0.4,
  hist_abusive_return_rate_before: 0.3,
  hist_chargebacks_before: 1,
  price_vs_category_avg: 1.8,
  days_to_return: 27,
};

const ACTION_LABELS: Record<ScoreResult["action"], string> = {
  approve: "APPROVE",
  manual_review: "MANUAL REVIEW",
  auto_decline: "AUTO DECLINE",
};

function NumberField({
  label,
  value,
  onChange,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <div>
      <label className="field-label" htmlFor={label}>
        {label}
      </label>
      <input
        id={label}
        type="number"
        step={step ?? 1}
        className="field-input mt-1"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="field-label" htmlFor={label}>
        {label}
      </label>
      <select id={label} className="field-input mt-1" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}

export function ScoreOrderTab() {
  const [order, setOrder] = useState<OrderInput>(DEFAULT_ORDER);
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function update<K extends keyof OrderInput>(key: K, value: OrderInput[K]) {
    setOrder((prev) => ({ ...prev, [key]: value }));
  }

  async function handleScore() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.scoreOrder(order);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to score order. Is the local API running?");
    } finally {
      setLoading(false);
    }
  }

  const scorePct = result ? Math.round(result.risk_score * 100) : 0;
  const tier = result ? tierFromScore(scorePct) : "review";
  const tierColor = `var(--color-tier-${tier})`;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
      <GlassCard glowColor="var(--color-accent-violet)" className="p-6 lg:col-span-3">
        <h2 className="font-display text-lg font-semibold">Score an order</h2>
        <p className="mb-6 text-sm text-gray-500">
          Enter a return's details to get a risk score and recommended action from the actual trained model.
        </p>

        <div className="grid grid-cols-2 gap-4">
          <NumberField label="Order value (Rs.)" value={order.order_value} onChange={(v) => update("order_value", v)} />
          <NumberField
            label="Price vs. category avg"
            value={order.price_vs_category_avg}
            step={0.1}
            onChange={(v) => update("price_vs_category_avg", v)}
          />
          <SelectField label="Category" value={order.category} options={CATEGORIES} onChange={(v) => update("category", v)} />
          <SelectField
            label="Payment method"
            value={order.payment_method}
            options={PAYMENT_METHODS}
            onChange={(v) => update("payment_method", v)}
          />
          <SelectField
            label="Return reason"
            value={order.return_reason}
            options={RETURN_REASONS}
            onChange={(v) => update("return_reason", v)}
          />
          <NumberField label="Days to return" value={order.days_to_return} onChange={(v) => update("days_to_return", v)} />
          <NumberField
            label="Customer's abusive-return rate"
            value={order.hist_abusive_return_rate_before}
            step={0.05}
            onChange={(v) => update("hist_abusive_return_rate_before", v)}
          />
          <NumberField
            label="Customer's chargeback count"
            value={order.hist_chargebacks_before}
            onChange={(v) => update("hist_chargebacks_before", v)}
          />
          <NumberField
            label="Account age (days)"
            value={order.account_age_days_at_order}
            onChange={(v) => update("account_age_days_at_order", v)}
          />
          <NumberField
            label="Orders before this one"
            value={order.hist_orders_before}
            onChange={(v) => update("hist_orders_before", v)}
          />
        </div>

        <button type="button" className="btn-primary mt-6 w-full" onClick={handleScore} disabled={loading}>
          {loading ? "Scoring…" : "Score this order"}
        </button>
        {error && <p className="mt-3 text-sm text-[var(--color-tier-decline)]">{error}</p>}
      </GlassCard>

      <GlassCard glowColor={tierColor} className="flex flex-col items-center p-6 lg:col-span-2">
        {result ? (
          <>
            <span className="field-label mb-4">Risk score</span>
            <Gauge value={scorePct} />
            <span className="verdict-badge mt-3" style={{ color: tierColor, background: `color-mix(in srgb, ${tierColor} 15%, transparent)` }}>
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: tierColor }} />
              {ACTION_LABELS[result.action]}
            </span>
            <p className="mt-4 text-center text-sm text-gray-300">{result.plain_english_summary}</p>
            <div className="mt-6 w-full">
              <div className="field-label mb-3">Top factors</div>
              <FactorList factors={result.top_factors} />
            </div>
          </>
        ) : (
          <div className="flex h-full flex-col items-center justify-center text-center text-sm text-gray-500">
            <p>Fill in the order details and click "Score this order" to see the model's decision.</p>
          </div>
        )}
      </GlassCard>
    </div>
  );
}
