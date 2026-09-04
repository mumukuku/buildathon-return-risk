/**
 * Sliding-pill tab navigation. The measured-position + spring-animation
 * technique is adapted from kokonutui's smooth-tab.tsx (MIT) -- we use just
 * that interaction pattern, not their full component (which bundles a large
 * animated "now playing"-style content panel we don't need; our tabs each
 * render their own full page content below the nav).
 */
import * as React from "react";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";

export interface TabNavItem {
  id: string;
  label: string;
}

export interface TabNavProps {
  items: TabNavItem[];
  selected: string;
  onChange: (id: string) => void;
  className?: string;
}

export function TabNav({ items, selected, onChange, className }: TabNavProps) {
  const buttonRefs = React.useRef<Map<string, HTMLButtonElement>>(new Map());
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = React.useState({ width: 0, left: 0 });

  React.useLayoutEffect(() => {
    const updateDimensions = () => {
      const selectedButton = buttonRefs.current.get(selected);
      const container = containerRef.current;
      if (selectedButton && container) {
        const rect = selectedButton.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        setDimensions({ width: rect.width, left: rect.left - containerRect.left });
      }
    };
    requestAnimationFrame(updateDimensions);
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, [selected]);

  return (
    <div
      ref={containerRef}
      role="tablist"
      aria-label="Dashboard sections"
      className={cn("relative flex gap-1 rounded-full p-1", className)}
    >
      <motion.div
        aria-hidden="true"
        className="absolute z-0 rounded-full"
        style={{ height: "calc(100% - 8px)", top: 4, background: "var(--color-accent-lime)" }}
        initial={false}
        animate={{ width: dimensions.width, x: dimensions.left }}
        transition={{ type: "spring", stiffness: 400, damping: 32 }}
      />
      {items.map((item) => {
        const isSelected = item.id === selected;
        return (
          <button
            key={item.id}
            ref={(el) => {
              if (el) buttonRefs.current.set(item.id, el);
              else buttonRefs.current.delete(item.id);
            }}
            role="tab"
            aria-selected={isSelected}
            type="button"
            onClick={() => onChange(item.id)}
            className={cn(
              "relative z-10 rounded-full px-4 py-2 text-sm font-medium transition-colors",
              isSelected ? "text-[#0A0A0C]" : "text-gray-400 hover:text-gray-200"
            )}
          >
            {item.label}
          </button>
        );
      })}
    </div>
  );
}
