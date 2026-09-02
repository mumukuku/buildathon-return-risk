import * as React from "react";
import { cn } from "@/lib/utils";
import { GlassFilter } from "@/components/kokonutui/glass-filter";

let glassIdCounter = 0;

export interface GlassCardProps extends React.ComponentProps<"div"> {
  /** Tailwind/CSS color for the corner glow accent, e.g. "var(--tier-review)" */
  glowColor?: string;
  glowSize?: number;
  liquidGlass?: boolean;
}

/**
 * Our dashboard's glass card. Structure follows kokonutui's LiquidGlassCard
 * (MIT) -- https://github.com/kokonut-labs/kokonutui -- with two deliberate
 * departures: no hover animation (static display, not an interactive
 * element) and a softened shadow (their original had a sharp bright inset
 * highlight line that read as a hard border rather than glass).
 *   1. A soft box-shadow (.liquid-glass-shadow in index.css): outer
 *      elevation + a diffuse inset glow, no defined edge line.
 *   2. A SEPARATE backdrop layer carrying the SVG displacement filter --
 *      we do NOT stack an extra large blur() on top of it; the filter's
 *      own internal feGaussianBlur stages already provide the blur, and
 *      over-blurring on top washes out the refraction pattern. The card's
 *      own subtle backdrop-blur-[2px] is the only extra blur.
 *   3. Content in its own stacking layer on top, so distortion never
 *      touches actual text/UI.
 * On top of their structure we layer our OWN accents: a per-card corner
 * glow tinted to that card's semantic color, and a confined grain texture.
 */
export function GlassCard({
  className,
  glowColor = "var(--color-accent-violet)",
  glowSize = 240,
  liquidGlass = true,
  children,
  ...props
}: GlassCardProps) {
  const filterId = React.useMemo(() => `glass-distortion-${glassIdCounter++}`, []);

  return (
    <div
      data-slot="glass-card"
      className={cn("relative isolate overflow-hidden rounded-2xl backdrop-blur-[2px]", className)}
      style={{ background: "rgba(255,255,255,0.05)" }}
      {...props}
    >
      {/* Layer 1: the real glass-edge shadow */}
      <div className="liquid-glass-shadow pointer-events-none absolute inset-0 rounded-[inherit]" />

      {/* Layer 2: SVG displacement filter on its own backdrop sublayer */}
      {liquidGlass && (
        <>
          <div
            className="pointer-events-none absolute inset-0 -z-10 overflow-hidden rounded-[inherit]"
            style={{ backdropFilter: `url("#${filterId}")`, WebkitBackdropFilter: `url("#${filterId}")` }}
          />
          <GlassFilter id={filterId} scale={30} />
        </>
      )}

      {/* Our own accents: corner glow + confined grain */}
      <div
        className="pointer-events-none absolute -right-16 -top-16 z-0 rounded-full"
        style={{ width: glowSize, height: glowSize, background: glowColor, filter: "blur(60px)", opacity: 0.3 }}
      />
      <div
        className="pointer-events-none absolute inset-0 z-0"
        style={{ filter: "url(#hatchGrainFilter)", opacity: 0.05, mixBlendMode: "overlay" }}
      />

      {/* Content, isolated from all distortion/effects above */}
      <div className="relative z-10">{children}</div>
    </div>
  );
}
