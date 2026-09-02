# Frontend TODO

## Planned rework: adopt component libraries

Once the current hand-built HTML/CSS preview design is locked, rebuild the
real frontend using:

- **bklit-ui** (https://github.com/bklit/bklit-ui) — shadcn/ui registry,
  Tailwind + Visx charts + Motion. Has a `RingChart` component ("multi-ring
  progress indicators with animated arcs") that should replace our
  hand-rolled SVG semicircle gauge — likely more robust and animatable.
  Install via: `npx shadcn@latest add @bklit/<component>`

- **kokonutui** (https://github.com/kokonut-labs/kokonutui) — shadcn/ui-based,
  Tailwind + Framer Motion, broader component set (cards, inputs, buttons,
  animated micro-interactions). Good candidate for polishing form fields,
  buttons, and card interactions.
  Install via: `npx shadcn@latest add https://kokonutui.com/r/<component>.json`

## Prerequisite

Both are shadcn/ui registries (copy-paste-into-project, not npm packages),
so the real React app needs shadcn/ui initialized first:
    npx shadcn@latest init

## Status

Design direction (glass cards, grain texture, diagonal hatch pattern, color
palette, gauge concept) already approved via HTML previews in
riskguard_preview/. This rework is about swapping the hand-rolled
implementation for these libraries' components where they fit, not
re-deciding the visual direction.

Not started yet -- noted for later in the build sequence.
