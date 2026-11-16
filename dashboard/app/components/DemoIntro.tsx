"use client";

import { useEffect, useState } from "react";
import { DEMO } from "../lib/demo";

/**
 * The first thirty seconds, for someone who has never seen this and has a queue behind them.
 *
 * Shown once per browser in the public build, then remembered. It does two jobs in one card:
 * it says what the project is, and it hands over four things to actually press, because a judge
 * who presses Start and watches their own finger drive the rep machine has understood more than
 * one who reads a paragraph about admittance control. The honest framing stays on the banner
 * behind it, which cannot be dismissed.
 */

const SEEN_KEY = "flexnsense.demo.intro.v1";

const TRY: { do: string; why: string }[] = [
  {
    do: "Press Start in the Rep machine, bottom left, and show your hand to the camera.",
    why: "Your finger drives it. The voice is the machine's own instruction table, the force and the arm are simulated.",
  },
  {
    do: "While a repetition is running, tick \"Simulate a lost force reading\".",
    why: "It stops. A missing force is never guessed and never held at the last value: that rule is the whole safety design.",
  },
  {
    do: "Ask the assistant: \"walk me through the dashboard\", or \"highlight the flexor tendons of the index finger\".",
    why: "It can move the camera, light up a tendon and scroll you to the panel it is talking about.",
  },
  {
    do: "Pick a week, then Export Medical Report.",
    why: "The recovery index, the per tendon fits and the report are the real calculations over a synthetic patient.",
  },
];

/**
 * Whether this browser has been here before. Read while the first render is decided rather than
 * from an effect, so the card is either there or not and never flashes into place. Safe because
 * this component is only ever mounted client-side (see DemoIntroClient).
 */
function firstVisit(): boolean {
  if (!DEMO) return false;
  try {
    return !window.localStorage.getItem(SEEN_KEY);
  } catch {
    return true; // storage blocked: showing it twice is better than never showing it
  }
}

export default function DemoIntro() {
  const [open, setOpen] = useState(firstVisit);

  const close = () => {
    setOpen(false);
    try {
      window.localStorage.setItem(SEEN_KEY, "1");
    } catch {
      /* nothing to remember it with, and nothing to do about that */
    }
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  if (!open) return null;
  return (
    <div
      className="absolute inset-0 z-30 flex items-center justify-center bg-black/70 p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="demo-intro-title"
      onClick={close}
    >
      <div
        className="panel max-h-full w-full max-w-[640px] space-y-4 overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="space-y-1">
          <h2 id="demo-intro-title" className="t-title">
            Flex &apos;N Sense: a force-limited finger tether
          </h2>
          <p className="t-body muted">
            After hand surgery a finger has to be moved early and often, with a force cap the surgeon sets. We
            built that from two robot arms, a 600 gf spring scale and a camera: the arm anchors the finger, and
            the moment the pull reaches the cap it gives way. This is the clinician&apos;s screen for it.
          </p>
        </div>

        <div
          className="rounded-md p-3 t-small"
          style={{ background: "rgba(251, 191, 36, 0.1)", border: "1px solid var(--warn)" }}
        >
          <span className="font-semibold">No hardware is connected to this page.</span>{" "}
          <span className="muted">
            No arm, no gauge. Every force number is simulated and no number here is a measurement. The hand
            tracking is real, using your own laptop camera in place of the camera over the patient&apos;s wrist.
            The notice at the top lists exactly what is real and what is not.
          </span>
        </div>

        <div className="space-y-2">
          <h3 className="t-section">Worth pressing</h3>
          <ol className="space-y-2">
            {TRY.map((t, i) => (
              <li key={t.do} className="flex gap-2.5">
                <span className="num faint shrink-0">{i + 1}</span>
                <span className="t-small">
                  <span className="t-body">{t.do}</span>
                  <br />
                  <span className="muted">{t.why}</span>
                </span>
              </li>
            ))}
          </ol>
        </div>

        <button className="btn btn-primary w-full" onClick={close} autoFocus>
          Open the dashboard
        </button>
      </div>
    </div>
  );
}
