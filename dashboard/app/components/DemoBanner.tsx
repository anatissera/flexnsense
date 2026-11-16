"use client";

import { useEffect, useRef, useState } from "react";
import { DEMO, type Hardware } from "../lib/demo";
import { fireEmergencyStop } from "../lib/estop";
import VoiceStop from "./VoiceStop";
import { HintsToggle } from "./Hints";
import { useHardwareStatus } from "./HardwareProvider";
import { INVENTORY, REALITY_LABEL, inventoryFor, type Reality } from "../lib/demoInventory";

/**
 * The standing statement of what this build is, as a bar above the dashboard.
 *
 * It is a bar in the layout rather than a floating panel because the viewer's corners are all
 * spoken for (controls top left, view cube top right, chat and rep machine bottom left, charts
 * down the right) and a notice that overlaps the view cube at 1440 px wide is a notice someone
 * will move out of the way. Taking 44 px off the top costs the dashboard nothing it was using.
 *
 * Two shapes, one component. Public build: an amber notice that nothing is connected, opening
 * into the full inventory. Bench build: the live status of the two local servers, because there
 * the useful fact is whether the gauge is running. It collapses but never dismisses: the claim
 * that no hardware is attached stays on screen for as long as it is true.
 */

const DOT: Record<Reality, string> = {
  real: "var(--phase-extension)",
  simulated: "var(--warn)",
  bench: "var(--ink-faint)",
};

const ORDER: Reality[] = ["real", "simulated", "bench"];

function Inventory() {
  return (
    <div className="absolute inset-x-0 top-full z-30 flex justify-center px-4">
      <div className="panel max-h-[70vh] w-full max-w-[760px] space-y-3 overflow-y-auto !rounded-t-none">
        {ORDER.map((reality) => (
          <section key={reality} className="space-y-1.5">
            <h4 className="t-section flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: DOT[reality] }} />
              {REALITY_LABEL[reality]}
            </h4>
            {inventoryFor(reality).map((item) => (
              <p key={item.what} className="t-small">
                <span className="font-semibold">{item.what}.</span>{" "}
                <span className="muted">{item.detail}</span>
              </p>
            ))}
          </section>
        ))}
        <p className="t-small faint border-t border-(--line) pt-2">
          {INVENTORY.length} entries. Flex &apos;N Sense, Leukerbad Robotics Hackathon 2026. For demonstration
          and research, not for clinical decisions.
        </p>
      </div>
    </div>
  );
}

/**
 * The one control that is always on screen.
 *
 * It aborts the running set through the same path the rep machine's own Stop uses. It cannot
 * reach an arm, and the line it leaves behind says so rather than letting a visitor believe the
 * browser is the thing holding the finger's safety.
 */
function StopButton() {
  const [stopped, setStopped] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);
  const fire = () => {
    fireEmergencyStop("stopped from the top bar");
    setStopped(true);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setStopped(false), 8000);
  };
  return (
    <>
      {stopped && (
        <span className="t-small shrink-0" style={{ color: "var(--danger)" }}>
          Stopped. A browser never moves an arm: on the rig the stop is the session server and the
          force gate, which release on their own if this screen goes away.
        </span>
      )}
      <button
        className="btn btn-danger btn-sm shrink-0"
        onClick={fire}
        title="Abort the running set and release the camera"
      >
        Stop session
      </button>
    </>
  );
}

function statusDot(status: Hardware[keyof Hardware]) {
  if (status === "present") return "var(--phase-extension)";
  if (status === "checking") return "var(--ink-faint)";
  return "var(--danger)";
}

function BenchBar({ hw }: { hw: Hardware }) {
  const row = (name: string, status: Hardware[keyof Hardware], hint: string) => (
    <span className="flex items-center gap-1.5" title={hint}>
      <span className="inline-block h-2 w-2 rounded-full" style={{ background: statusDot(status) }} />
      {name} {status === "present" ? "live" : status === "checking" ? "checking…" : "not running"}
    </span>
  );
  return (
    <header className="flex h-11 shrink-0 items-center gap-4 border-b border-(--line) px-4 t-small">
      <span className="t-section">Flex &apos;N Sense</span>
      <span className="muted">Bench build</span>
      {row("Gauge reader", hw.gauge, "python -m gauge_reader.server on 127.0.0.1:8766")}
      {row("WiLoR tracker", hw.wilor, "python -m hand_tracking.server on 127.0.0.1:8765")}
      <span className="ml-auto flex items-center gap-2">
        <VoiceStop />
        <StopButton />
        <HintsToggle />
      </span>
    </header>
  );
}

export default function DemoBanner() {
  const hw = useHardwareStatus();
  const [open, setOpen] = useState(false);
  if (!DEMO) return <BenchBar hw={hw} />;
  return (
    <header
      className="relative flex h-11 shrink-0 items-center gap-3 border-b px-4"
      style={{ borderColor: "var(--warn)", background: "rgba(251, 191, 36, 0.08)" }}
      role="note"
    >
      <span aria-hidden className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: "var(--warn)" }} />
      <span className="t-section shrink-0">Flex &apos;N Sense</span>
      <p className="t-small min-w-0 truncate">
        <span className="font-semibold">Demonstration build: no robot arm and no force gauge are connected.</span>{" "}
        <span className="muted">
          Every force number is simulated. The hand tracking is real and uses your own laptop camera.
        </span>
      </p>
      <span className="ml-auto flex shrink-0 items-center gap-2">
        <VoiceStop />
        <StopButton />
        <HintsToggle />
        <button className="btn btn-sm" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? "Hide" : "What is real here?"}
        </button>
      </span>
      {open && <Inventory />}
    </header>
  );
}
