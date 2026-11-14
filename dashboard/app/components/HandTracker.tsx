"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import { buildHandFrame, FINGERTIPS, type HandFrame } from "../lib/handFrame";

/** Joint name -> angle (rad) for the MyoHand model, written every frame while tracking. */
export type HandPose = Record<string, number>;

type P3 = { x: number; y: number; z: number };

const sub = (a: P3, b: P3): P3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
/** Angle between two bone vectors: 0 when the bones are collinear (straight joint), grows with flexion. */
function bend(u: P3, v: P3): number {
  const du = Math.hypot(u.x, u.y, u.z);
  const dv = Math.hypot(v.x, v.y, v.z);
  if (du < 1e-6 || dv < 1e-6) return 0;
  const c = (u.x * v.x + u.y * v.y + u.z * v.z) / (du * dv);
  return Math.acos(Math.min(Math.max(c, -1), 1));
}

/**
 * MediaPipe world landmarks (21 points) -> MyoHand flexion joints.
 * Fingers 2-5: MCP = wrist->MCP vs proximal phalanx, PIP = proximal vs middle, DIP = middle vs distal.
 * Thumb: CMC, MP and IP flexion (all negative in flexion in MyoHand, verified headlessly by tip-to-palm distance).
 */
export function landmarksToPose(w: P3[]): HandPose {
  const pose: HandPose = {};
  const bones = (i: number) => sub(w[i + 1], w[i]);
  [2, 3, 4, 5].forEach((d, k) => {
    const b = 5 + 4 * k; // MCP landmark index: 5, 9, 13, 17
    pose[`mcp${d}_flexion_r`] = bend(sub(w[b], w[0]), bones(b));
    pose[`pm${d}_flexion_r`] = bend(bones(b), bones(b + 1));
    pose[`md${d}_flexion_r`] = bend(bones(b + 1), bones(b + 2));
  });
  pose["cmc_flexion_r"] = -bend(bones(0), bones(1));
  pose["mp_flexion_r"] = -bend(bones(1), bones(2));
  pose["ip_flexion_r"] = -bend(bones(2), bones(3));
  return pose;
}

const dist = (a: P3, b: P3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
/**
 * Index displacement in mm: how much the wrist-to-fingertip keypoint distance has shrunk relative to
 * `ref` (the same distance with the finger extended). World landmarks are metric, so this is real mm and
 * does not depend on hand rotation or camera distance.
 */
export const tipWristMm = (w: P3[]) => dist(w[8], w[0]) * 1000;

const SMOOTH = 0.5; // EMA weight of the new sample
const TIPS = new Set<number>(FINGERTIPS); // drawn larger and in a second colour

export default function HandTracker({ pose, disp, onStart, onActive, onFrame }: {
  pose: RefObject<HandPose | null>;
  /** Written every tracked frame with the index tip displacement in mm (null when not tracking). */
  disp?: RefObject<number | null>;
  onStart?: () => void;
  /** Called when hand tracking is switched on or off. */
  onActive?: (on: boolean) => void;
  /**
   * Called once per frame a hand is found, with the full tracked state in the shared HandFrame
   * shape (see app/lib/handFrame.ts). A WiLoR backend emits the same shape, so a consumer here
   * does not need to know which model produced it.
   */
  onFrame?: (frame: HandFrame) => void;
}) {
  const [on, setOn] = useState(false);
  const [status, setStatus] = useState("");
  const [dispMm, setDispMm] = useState<number | null>(null);
  const tipRef = useRef<{ now: number; ref: number | null }>({ now: 0, ref: null });
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  // Held in a ref, and declared before the effect that reads it, so that a caller passing a fresh
  // closure on every render does not tear down and restart the camera.
  const onFrameRef = useRef(onFrame);
  useEffect(() => { onFrameRef.current = onFrame; });

  useEffect(() => {
    if (!on) return;
    let cancelled = false;
    let raf = 0;
    let stream: MediaStream | null = null;
    let landmarker: { close: () => void } | null = null;
    const smoothed: HandPose = {};

    (async () => {
      try {
        setStatus("Loading hand model…");
        const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
        const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
        const make = (delegate: "GPU" | "CPU") =>
          HandLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: "/mediapipe/hand_landmarker.task", delegate },
            runningMode: "VIDEO",
            numHands: 1,
          });
        const hl = await make("GPU").catch(() => make("CPU"));
        if (cancelled) { hl.close(); return; }
        landmarker = hl;

        setStatus("Starting camera…");
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play();
        setStatus("Show your hand to the camera");

        let lastTime = -1;
        let lastSeen = 0;
        const t0 = performance.now();
        const loop = () => {
          raf = requestAnimationFrame(loop);
          if (video.readyState < 2 || video.currentTime === lastTime) return;
          lastTime = video.currentTime;
          const res = hl.detectForVideo(video, performance.now());
          const lm = res.landmarks[0];
          const cv = overlayRef.current;
          const ctx = cv?.getContext("2d");
          if (cv && ctx) {
            cv.width = video.videoWidth;
            cv.height = video.videoHeight;
            ctx.clearRect(0, 0, cv.width, cv.height);
            (lm ?? []).forEach((p, i) => {
              const isTip = TIPS.has(i);
              ctx.fillStyle = isTip ? "#ffc14d" : "#4df5e8";
              ctx.beginPath();
              ctx.arc(p.x * cv.width, p.y * cv.height, isTip ? 6 : 3, 0, 2 * Math.PI);
              ctx.fill();
            });
          }
          const w = res.worldLandmarks[0];
          if (w) {
            const next = landmarksToPose(w);
            for (const [k, v] of Object.entries(next)) smoothed[k] = k in smoothed ? smoothed[k] + SMOOTH * (v - smoothed[k]) : v;
            pose.current = { ...smoothed };
            const t = tipRef.current;
            t.now = t.now ? t.now + SMOOTH * (tipWristMm(w) - t.now) : tipWristMm(w);
            t.ref = t.ref === null ? t.now : Math.max(t.ref, t.now); // reference = most extended so far
            setDispMm(t.ref - t.now);
            if (disp) disp.current = t.ref - t.now;
            if (lm && onFrameRef.current) {
              const side = res.handedness?.[0]?.[0]?.categoryName?.toLowerCase();
              // Angles and tipWristMm are the smoothed values, the same ones the 3D hand follows.
              onFrameRef.current(buildHandFrame({
                source: "mediapipe",
                t: performance.now() - t0,
                handedness: side === "left" || side === "right" ? side : null,
                landmarks: lm,
                world: w,
                angles: smoothed,
                tipWristMm: t.now,
              }));
            }
            lastSeen = performance.now();
            setStatus("Tracking");
          } else if (performance.now() - lastSeen > 800) {
            setStatus("No hand detected (holding last pose)");
          }
        };
        raf = requestAnimationFrame(loop);
      } catch (e) {
        console.error(e);
        if (!cancelled) {
          setStatus(`Hand tracking failed: ${e instanceof Error ? e.message : String(e)}`);
          setOn(false);
        }
      }
    })();

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
      landmarker?.close();
      pose.current = null;
      if (disp) disp.current = null;
      tipRef.current = { now: 0, ref: null };
      setDispMm(null);
    };
  }, [on, pose, disp]);

  const onActiveRef = useRef(onActive);
  useEffect(() => { onActiveRef.current = onActive; });
  useEffect(() => {
    onActiveRef.current?.(on);
    return () => { if (on) onActiveRef.current?.(false); };
  }, [on]);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <button
          className={on ? "btn btn-danger" : "btn"}
          aria-pressed={on}
          onClick={() => { if (!on) onStart?.(); setOn(!on); if (on) setStatus(""); }}
        >
          {on ? "Stop hand tracking" : "Hand tracking mode"}
        </button>
        {status && <span className="t-small muted">{status}</span>}
      </div>
      {on && (
        <>
          <div className="relative w-48 overflow-hidden rounded" style={{ transform: "scaleX(-1)" }}>
            <video ref={videoRef} muted playsInline className="block w-full" />
            <canvas ref={overlayRef} className="absolute inset-0 h-full w-full" />
          </div>
          <div className="flex items-center gap-2 t-body">
            <span>Index displacement</span>
            <span className="num font-semibold">{dispMm === null ? "–" : dispMm.toFixed(1)} mm</span>
            <button className="btn" onClick={() => { tipRef.current.ref = tipRef.current.now || null; }}>Zero (finger extended)</button>
          </div>
          <p className="t-small muted">
            Camera frames are processed in the browser and never stored or uploaded. The 3D hand mirrors your finger flexion.
            Amber dots are the five fingertips, cyan the other tracked joints.
          </p>
        </>
      )}
    </div>
  );
}
