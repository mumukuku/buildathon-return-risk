/**
 * Vendored from kokonut-labs/kokonutui (liquid-glass-card.tsx), MIT licensed.
 * https://github.com/kokonut-labs/kokonutui
 *
 * We extract just the SVG displacement-filter technique (the actual "liquid
 * glass" refraction effect) rather than their full carousel/media card --
 * that component is a specific media showcase (next/image, play/pause,
 * carousel arrows) that doesn't fit our generic dashboard cards. This filter
 * is the genuinely reusable part: apply `filter: url(#glass-distortion)` to
 * any element for the same frosted-glass refraction look.
 */
import React from "react";

interface GlassFilterProps {
  id: string;
  scale?: number;
}

export const GlassFilter = React.memo(
  ({ id, scale = 60 }: GlassFilterProps) => (
    <svg aria-hidden="true" className="hidden" focusable={false}>
      <title>Glass Effect Filter</title>
      <defs>
        <filter
          colorInterpolationFilters="sRGB"
          height="200%"
          id={id}
          width="200%"
          x="-50%"
          y="-50%"
        >
          <feTurbulence
            baseFrequency="0.05 0.05"
            numOctaves={1}
            result="turbulence"
            seed={1}
            type="fractalNoise"
          />
          <feGaussianBlur in="turbulence" result="blurredNoise" stdDeviation={2} />
          <feDisplacementMap
            in="SourceGraphic"
            in2="blurredNoise"
            result="displaced"
            scale={scale}
            xChannelSelector="R"
            yChannelSelector="B"
          />
          <feGaussianBlur in="displaced" result="finalBlur" stdDeviation={4} />
          <feComposite in="finalBlur" in2="finalBlur" operator="over" />
        </filter>
      </defs>
    </svg>
  )
);
GlassFilter.displayName = "GlassFilter";
