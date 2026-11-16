"use client";

import { useEffect, useState } from "react";

/**
 * Says out loud that this is a laptop screen's dashboard, instead of letting a phone discover it.
 *
 * The layout is a 3D view with instrument panels anchored to its corners, two camera previews
 * side by side and a 470 px column of charts. That is a clinician's screen and it does not fold
 * into a phone: shrinking it would not make it usable, it would make it a different product. So
 * a small viewport gets an honest card rather than a broken page, with a way through for anyone
 * who wants to look anyway.
 */

/**
 * What the layout actually needs, measured rather than guessed: the left column stacks the
 * viewer's controls, the rep machine and the chat at fixed heights and wants about 850 px, and
 * the right column of charts is 470 px next to a 3D view worth looking at.
 */
const MIN_W = 1280;
const MIN_H = 800;

export default function ScreenGate({ children }: { children: React.ReactNode }) {
  const [small, setSmall] = useState(false);
  const [override, setOverride] = useState(false);

  useEffect(() => {
    const check = () => setSmall(window.innerWidth < MIN_W || window.innerHeight < MIN_H);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  if (!small || override) return <>{children}</>;
  return (
    <div className="flex min-h-0 w-full flex-1 items-center justify-center p-6">
      <div className="panel w-full max-w-[420px] space-y-3">
        <div className="t-title">Flex &apos;N Sense</div>
        <p className="t-body">
          This dashboard is built for a laptop screen. It is a clinician&apos;s instrument panel: a 3D hand with
          readouts anchored around it, two camera previews and a column of charts, all on screen at once.
        </p>
        <p className="t-small muted">
          Open it on a computer at least {MIN_W} by {MIN_H} pixels. The hand tracking needs a webcam there too,
          which is the part worth trying.
        </p>
        <p className="t-small faint">
          Demonstration build: no robot arm and no force gauge are connected, and every force number is simulated.
        </p>
        <button className="btn btn-primary w-full" onClick={() => setOverride(true)}>
          Show it anyway
        </button>
      </div>
    </div>
  );
}
