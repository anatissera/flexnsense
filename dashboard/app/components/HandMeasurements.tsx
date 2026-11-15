"use client";

import { useEffect, useRef, useState } from "react";
import { ExcursionRecorder, type ExcursionResult, type Point, type TrackedFinger } from "../lib/excursion";
import { allFlexionDeg, type FingerName } from "../lib/handGeometry";
import type { HandFrame } from "../lib/handFrame";

const NAMES: FingerName[] = ["thumb", "index", "middle", "ring", "pinky"];
/** Bar scale. PIP flexion past this is rare in a healthy hand. */
const BAR_MAX_DEG = 120;
const SMOOTH = 0.5;
/** A hand not seen for this long is shown as lost rather than frozen at its last value. */
const STALE_MS = 800;
/**
 * Below this, the sweep was not close to one straight line (a flinch, a rotated wrist, a second
 * finger moving), so the travel is a line fitted through something that was not 1-D motion. A
 * heuristic cut-off, not a clinical one: the docs only say a clean pull is a few percent off
 * axis, and that the number is untrustworthy as it approaches the travel itself.
 */
const MIN_VARIANCE_EXPLAINED = 0.8;

type Subscribe = (fn: (frame: HandFrame) => void) => () => void;

const toPoints = (world: HandFrame["world"]): Point[] => world.map((p) => [p.x, p.y, p.z]);

/**
 * Live per-finger flexion in degrees, and a range-of-motion sweep: move one finger slowly from
 * straight to fully bent and read how far its tip travelled along its own path.
 *
 * Ports of `hand_tracking/landmarks.py` and `hand_tracking/excursion.py`, fed from the same
 * `HandFrame` stream the 3D hand uses, so it works with either MediaPipe or WiLoR.
 */
export default function HandMeasurements({ subscribe }: { subscribe: Subscribe }) {
  const [degrees, setDegrees] = useState<Partial<Record<FingerName, number>> | null>(null);
  const [sweepFinger, setSweepFinger] = useState<TrackedFinger>("index");
  const [recording, setRecording] = useState(false);
  const [frames, setFrames] = useState(0);
  const [result, setResult] = useState<ExcursionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const smoothed = useRef<Partial<Record<FingerName, number>>>({});
  const lastSeen = useRef(0);
  const recorder = useRef<ExcursionRecorder | null>(null);

  useEffect(() => {
    const unsubscribe = subscribe((frame) => {
      const points = toPoints(frame.world);
      const now = allFlexionDeg(points);
      for (const name of NAMES) {
        const v = now[name];
        if (!Number.isFinite(v)) continue;
        const prev = smoothed.current[name];
        smoothed.current[name] = prev === undefined ? v : prev + SMOOTH * (v - prev);
      }
      lastSeen.current = performance.now();
      recorder.current?.add(points);
    });
    // Poll for display so a 30 fps stream does not re-render the panel 30 times a second.
    const timer = window.setInterval(() => {
      const fresh = performance.now() - lastSeen.current < STALE_MS;
      setDegrees(fresh ? { ...smoothed.current } : null);
      setFrames(recorder.current?.length ?? 0);
    }, 100);
    return () => {
      unsubscribe();
      window.clearInterval(timer);
    };
  }, [subscribe]);

  const start = () => {
    recorder.current = new ExcursionRecorder(sweepFinger);
    setResult(null);
    setError(null);
    setRecording(true);
  };

  const stop = () => {
    setRecording(false);
    const rec = recorder.current;
    recorder.current = null;
    try {
      if (!rec) throw new Error("nothing was recorded");
      setResult(rec.result());
    } catch (e) {
      setError(
        rec && rec.length < 2
          ? "Too few frames. Keep your hand in view during the sweep."
          : e instanceof Error ? e.message : String(e),
      );
    }
  };

  const travelMm = result ? result.excursionM * 1000 : null;
  const offAxisMm = result ? result.perpendicularRmsM * 1000 : null;
  const clean = result ? result.varianceExplained >= MIN_VARIANCE_EXPLAINED : false;

  return (
    <div className="space-y-2 rounded border border-(--line) p-2">
      <div className="t-section">Hand measurements</div>

      <div className="space-y-1">
        {NAMES.map((name) => {
          const v = degrees?.[name];
          return (
            <div key={name} className="flex items-center gap-2 t-small">
              <span className="w-12 capitalize muted">{name}</span>
              <div className="relative h-2 flex-1 rounded-full bg-(--surface-raised)">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${v === undefined ? 0 : Math.min(Math.max(v / BAR_MAX_DEG, 0), 1) * 100}%`,
                    background: "#4df5e8",
                  }}
                />
              </div>
              <span className="num w-12 text-right font-semibold">{v === undefined ? "–" : `${v.toFixed(0)}°`}</span>
            </div>
          );
        })}
        <p className="t-small faint">
          {degrees === null
            ? "No hand in view."
            : "Flexion at the middle joint (PIP; the thumb at its IP joint). 0° is straight."}
        </p>
      </div>

      <div className="space-y-1 border-t border-(--line) pt-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="t-small muted">Range of motion</span>
          <select
            className="btn btn-sm"
            value={sweepFinger}
            disabled={recording}
            onChange={(e) => setSweepFinger(e.target.value as TrackedFinger)}
          >
            {NAMES.map((n) => <option key={n} value={n} className="text-black">{n}</option>)}
          </select>
          {recording ? (
            <button className="btn btn-sm btn-danger" onClick={stop}>Stop sweep</button>
          ) : (
            <button className="btn btn-sm btn-primary" onClick={start} disabled={degrees === null}>Start sweep</button>
          )}
          {recording && <span className="t-small muted num">{frames} frames</span>}
        </div>
        {recording && (
          <p className="t-small muted">Move the {sweepFinger} finger slowly from straight to fully bent, then press Stop.</p>
        )}
        {error && <p className="t-small" style={{ color: "var(--danger)" }}>{error}</p>}
        {result && travelMm !== null && offAxisMm !== null && (
          <div className="space-y-1">
            <div className="flex items-baseline gap-2">
              <span className="t-small muted capitalize">{sweepFinger} tip travel</span>
              <span className="num text-lg font-semibold">{travelMm.toFixed(1)} mm</span>
            </div>
            <div className="t-small muted num">
              {(result.varianceExplained * 100).toFixed(0)}% of the motion along one line · {offAxisMm.toFixed(1)} mm off-axis · {result.nSamples} frames
            </div>
            {!clean && (
              <p className="t-small" style={{ color: "var(--warn)" }}>
                The motion was not close to a straight line (a wrist rotation or another finger moving?), so this travel is unreliable. Repeat the sweep.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
