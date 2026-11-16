"use client";

/**
 * Which of the two builds this is, and what it is allowed to claim.
 *
 * There is one dashboard and two ways it runs:
 *
 *   NEXT_PUBLIC_DEMO=1   the public demonstration build (Vercel). No bench is reachable from
 *                        it, by construction: the gauge reader and the WiLoR tracker listen on
 *                        127.0.0.1 of whoever is running them, which is not the visitor. So it
 *                        does not probe, it says plainly that nothing is connected, and every
 *                        panel that would show a measurement shows a simulation instead.
 *
 *   unset                the bench build (localhost, next to the rig). It probes the two local
 *                        servers and reports what it actually found, so the same screen can
 *                        say "simulated" in the morning and "live" once the gauge is running.
 *
 * The flag decides the copy, the probe decides the state. Neither ever invents a measurement:
 * a panel with no source says so, exactly as the control loop treats a missing force reading.
 */
import { useEffect, useState } from "react";
import { GAUGE_URL } from "../components/GaugeForce";
import { WILOR_URL } from "../components/HandTracker";

/** True in the public demonstration build. Inlined by Next at build time. */
export const DEMO = process.env.NEXT_PUBLIC_DEMO === "1";

/**
 * `unknown` is the demo build, which never looks. The bench build moves from `checking` to
 * `present` or `absent` and keeps checking, so starting a server mid-session lights the badge.
 */
export type HardwareStatus = "unknown" | "checking" | "present" | "absent";

export type Hardware = {
  /** `python -m hand_tracking.server`, the WiLoR GPU tracker. */
  wilor: HardwareStatus;
  /** `python -m gauge_reader.server`, the camera that reads the PESOLA. */
  gauge: HardwareStatus;
};

const PROBE_TIMEOUT_MS = 2000;
const PROBE_EVERY_MS = 8000;

/** Open the socket, see whether it opens, close it again. Nothing is sent. */
function probe(url: string): Promise<boolean> {
  return new Promise((resolve) => {
    let settled = false;
    const done = (ok: boolean) => {
      if (settled) return;
      settled = true;
      resolve(ok);
    };
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      done(false);
      return;
    }
    const timer = setTimeout(() => { done(false); ws.close(); }, PROBE_TIMEOUT_MS);
    ws.onopen = () => { clearTimeout(timer); done(true); ws.close(); };
    ws.onerror = () => { clearTimeout(timer); done(false); };
    ws.onclose = () => { clearTimeout(timer); done(false); };
  });
}

/**
 * What is reachable from this browser. In the demo build this is a constant and no socket is
 * ever opened: probing a visitor's own machine from a public page would be both pointless and
 * rude, and in some browsers a mixed-content error in their console.
 */
export function useHardware(): Hardware {
  const [state, setState] = useState<Hardware>(
    DEMO ? { wilor: "unknown", gauge: "unknown" } : { wilor: "checking", gauge: "checking" },
  );
  useEffect(() => {
    if (DEMO) return;
    let cancelled = false;
    const round = async () => {
      const [wilor, gauge] = await Promise.all([probe(WILOR_URL), probe(GAUGE_URL)]);
      if (!cancelled) setState({ wilor: wilor ? "present" : "absent", gauge: gauge ? "present" : "absent" });
    };
    round();
    const id = setInterval(round, PROBE_EVERY_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, []);
  return state;
}

/** True when the force gauge is genuinely attached and this build is allowed to say so. */
export const gaugeIsLive = (hw: Hardware) => !DEMO && hw.gauge === "present";
