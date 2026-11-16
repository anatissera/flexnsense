"use client";

import { useSyncExternalStore, type ReactNode } from "react";

/**
 * What each control does, written next to the control, for someone who has thirty seconds.
 *
 * A tooltip is the usual answer and it is the wrong one here: it needs a mouse to hover, it shows
 * one thing at a time, and a visitor who does not know a control exists never hovers it. These
 * notes are simply visible, on by default for a first visit, and switched off by anyone who finds
 * them in the way. They are where the "no hardware is attached, and here is what that means for
 * THIS button" sentences live, because a general disclaimer in a bar does not answer "so what
 * happens if I press this".
 *
 * The on/off lives in a module store read through `useSyncExternalStore` rather than in state or
 * a context. The value comes from localStorage, which the server cannot see: the server snapshot
 * is "on", the browser's is whatever was chosen last, and this is the one hook that is allowed to
 * disagree across hydration without it being a bug.
 */

const HINTS_KEY = "flexnsense.demo.hints.v1";

let cached: boolean | null = null;
const listeners = new Set<() => void>();

function read(): boolean {
  if (cached !== null) return cached;
  try {
    const raw = window.localStorage.getItem(HINTS_KEY);
    cached = raw === null ? true : raw === "1"; // on until somebody turns them off
  } catch {
    cached = true;
  }
  return cached;
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

function toggleHints(): void {
  cached = !read();
  try {
    window.localStorage.setItem(HINTS_KEY, cached ? "1" : "0");
  } catch {
    /* the choice just will not survive a reload */
  }
  listeners.forEach((fn) => fn());
}

export const useHints = () => useSyncExternalStore(subscribe, read, () => true);

export function HintsToggle() {
  const on = useHints();
  return (
    <button
      className={on ? "btn btn-sm btn-primary" : "btn btn-sm"}
      aria-pressed={on}
      onClick={toggleHints}
      title="Show or hide the notes that say what each control does without hardware attached"
    >
      Hints: {on ? "on" : "off"}
    </button>
  );
}

/**
 * One note. `kind` decides the colour: "demo" for the ones whose whole point is that there is no
 * hardware, "info" for the ones that would be just as true at the bench.
 */
export function Hint({ children, kind = "info" }: { children: ReactNode; kind?: "info" | "demo" }) {
  if (!useHints()) return null;
  return (
    <p
      className="rounded-r-md border-l-2 py-1 pl-2 t-small"
      style={{
        borderColor: kind === "demo" ? "var(--warn)" : "var(--ink-faint)",
        background: kind === "demo" ? "rgba(251, 191, 36, 0.07)" : "var(--line-soft)",
        color: "var(--ink-muted)",
      }}
    >
      {children}
    </p>
  );
}
