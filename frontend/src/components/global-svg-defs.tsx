/**
 * Global SVG filter/pattern definitions, rendered once at the app root.
 * Referenced elsewhere via `filter: url(#hatchGrainFilter)` and
 * `fill: url(#hatchPattern)` / `stroke: url(#hatchPattern)`.
 *
 * NOTE on flip + rotate: for a plain diagonal LINE texture (not an
 * asymmetric shape), mirroring it and rotating it 90 degrees produce the
 * SAME visual result -- both just swap the line between "/" and "\".
 * Applying them as two separate additive transforms cancels out back to
 * the original direction. So this applies a single 90-degree rotation,
 * which satisfies both asks at once rather than redundantly stacking them.
 *
 * The hatch uses bklit-ui's PatternLines (backed by @visx/pattern) for the
 * actual line-drawing. Rotation needs a small extra wrapper: @visx/pattern's
 * underlying `Pattern` component is hardcoded to only accept
 * id/width/height/children and does not forward a patternTransform prop, so
 * passing one directly to PatternLines is silently dropped. Instead we
 * render their pattern as a base tile, then reference it via
 * fill="url(#...)" inside one thin outer <pattern> that carries the actual
 * working patternTransform="rotate(90)".
 */
import { PatternLines } from "@/components/charts/visx-pattern";

export function GlobalSvgDefs() {
  return (
    <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true">
      <defs>
        <filter id="hatchGrainFilter">
          <feTurbulence type="fractalNoise" baseFrequency={0.9} numOctaves={2} stitchTiles="stitch" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <pattern id="hatchPattern" width={9} height={9} patternUnits="userSpaceOnUse" patternTransform="rotate(90)">
          <rect width={9} height={9} fill="url(#hatchPatternBase)" />
        </pattern>
      </defs>
      <PatternLines
        id="hatchPatternBase"
        height={9}
        width={9}
        strokeWidth={4}
        stroke="rgba(190,192,200,0.55)"
        background="rgba(255,255,255,0.10)"
        orientation={["diagonal"]}
      />
    </svg>
  );
}
