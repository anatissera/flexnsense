"use client";

/**
 * The one-word answer to "where is this number coming from", sitting on the panel itself.
 *
 * A visitor should never have to remember the banner to know what they are looking at, so every
 * panel that produces a number carries one of these. The colours are the ones the dashboard
 * already uses for state: amber for a simulation, the flexion blue for the visitor's own camera,
 * green for a reading that is genuinely live, and a quiet outline for what is not here at all.
 */

export type BadgeKind = "simulated" | "webcam" | "bench" | "live";

const STYLE: Record<BadgeKind, { label: string; bg: string; ink: string; title: string }> = {
  simulated: {
    label: "Simulated",
    bg: "var(--warn)",
    ink: "#0b0d11",
    title: "These numbers come from a model, not from a sensor",
  },
  webcam: {
    label: "Your camera",
    bg: "var(--phase-flexion)",
    ink: "#0b0d11",
    title: "Running in this browser on your own camera. Frames are measured and discarded",
  },
  bench: {
    label: "Needs the bench",
    bg: "transparent",
    ink: "var(--ink-muted)",
    title: "This panel needs the hardware at the rig and cannot work from a public page",
  },
  live: {
    label: "Live",
    bg: "var(--phase-extension)",
    ink: "#0b0d11",
    title: "A real reading from the hardware attached to this machine",
  },
};

export default function DemoBadge({ kind, label }: { kind: BadgeKind; label?: string }) {
  const s = STYLE[kind];
  return (
    <span
      className="tag"
      title={s.title}
      style={{
        background: s.bg,
        color: s.ink,
        border: kind === "bench" ? "1px solid var(--line)" : undefined,
      }}
    >
      {label ?? s.label}
    </span>
  );
}
