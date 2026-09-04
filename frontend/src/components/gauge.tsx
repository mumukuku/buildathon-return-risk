import { RingChart } from "@/components/charts/ring-chart";
import { Ring } from "@/components/charts/ring";
import { RingCenter } from "@/components/charts/ring-center";

export type RiskTier = "approve" | "review" | "decline";

const TIER_COLORS: Record<RiskTier, string> = {
  approve: "var(--color-tier-approve)",
  review: "var(--color-tier-review)",
  decline: "var(--color-tier-decline)",
};

export function tierFromScore(score0to100: number): RiskTier {
  if (score0to100 >= 75) return "decline";
  if (score0to100 >= 33) return "review";
  return "approve";
}

export interface GaugeProps {
  /** 0-100 */
  value: number;
  size?: number;
}

/**
 * Full-circle risk gauge built on bklit-ui's RingChart -- see
 * frontend/TODO.md for why we use their ring-chart rather than their
 * segmented notch-style gauge-chart.
 *
 * We deliberately use the component's DEFAULT angles (full circle) rather
 * than fighting to derive their exact semicircle angle convention -- a
 * clean full ring reads just as well for a single score and sidesteps a
 * geometry rabbit hole with no payoff.
 *
 * Hover interaction (built into their Ring component with no prop to
 * disable it) is suppressed via `pointer-events-none` -- this is a static
 * score display, not an interactive chart.
 *
 * The hatch texture on the unfilled track comes from overriding `--border`
 * (the CSS variable bklit's Ring component fills the background arc with)
 * to our `url(#hatchPattern)` SVG pattern defined in GlobalSvgDefs -- no
 * forking of their component source needed.
 */
export function Gauge({ value, size = 220 }: GaugeProps) {
  const tier = tierFromScore(value);
  const color = TIER_COLORS[tier];

  return (
    <div className="pointer-events-none" style={{ ["--border" as string]: "url(#hatchPattern)" }}>
      <RingChart
        data={[{ label: "Risk", value, maxValue: 100, color }]}
        size={size}
        strokeWidth={26}
        baseInnerRadius={70}
        enterTransition={{ type: "spring", stiffness: 300, damping: 26 }}
      >
        <Ring index={0} lineCap="round" />
        <RingCenter suffix="%" defaultLabel="" />
      </RingChart>
    </div>
  );
}
