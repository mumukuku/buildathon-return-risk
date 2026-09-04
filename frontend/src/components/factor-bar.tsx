export interface FactorBarProps {
  label: string;
  impact: number;
  maxAbsImpact: number;
}

export function FactorBar({ label, impact, maxAbsImpact }: FactorBarProps) {
  const color = impact >= 0 ? "var(--color-tier-decline)" : "var(--color-tier-approve)";
  const widthPct = maxAbsImpact > 0 ? Math.round((Math.abs(impact) / maxAbsImpact) * 100) : 0;

  return (
    <div className="mb-3 grid grid-cols-[1fr_90px] items-center gap-3 last:mb-0">
      <div>
        <div className="text-sm">{label}</div>
        <div className="factor-track mt-1">
          <div className="factor-fill" style={{ width: `${widthPct}%`, background: color }} />
        </div>
      </div>
      <div className="text-right font-mono text-xs" style={{ color }}>
        {impact >= 0 ? "+" : ""}
        {impact.toFixed(2)}
      </div>
    </div>
  );
}

export function FactorList({ factors }: { factors: { feature: string; impact: number }[] }) {
  const maxAbsImpact = Math.max(...factors.map((f) => Math.abs(f.impact)), 0.0001);
  return (
    <div>
      {factors.map((f) => (
        <FactorBar key={f.feature} label={f.feature} impact={f.impact} maxAbsImpact={maxAbsImpact} />
      ))}
    </div>
  );
}
