"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import CameraSelect, { useCameraDevices, videoConstraints } from "./CameraSelect";

/** One reading from the gauge reader. `n` is newtons (the unit the dashboard runs on). */
export type GaugeSample = { n: number; g: number; stable: boolean; at: number };

/** Where `python -m gauge_reader.server` listens by default. */
export const GAUGE_URL = "ws://127.0.0.1:8766/force";
/** A reading older than this is treated as no reading: never hold the last value over a dropout. */
export const GAUGE_STALE_MS = 500;
const JPEG_QUALITY = 0.8;

/** The sample if it is fresh, else null. Consumers call this rather than reading the ref raw. */
export function freshGauge(ref: RefObject<GaugeSample | null>, now = performance.now()): GaugeSample | null {
  const s = ref.current;
  return s && now - s.at <= GAUGE_STALE_MS ? s : null;
}

/**
 * Pick which camera watches the PESOLA scale, and stream its frames to the local gauge server. The server
 * measures each frame and discards it (nothing is written to disk) and sends back newtons, which land in
 * `force`. When there is no valid reading the ref is set to null, so a lost marker is "no force", not the
 * last number.
 */
export default function GaugeForce({ force, onActive }: {
  force: RefObject<GaugeSample | null>;
  /** Called when force sensing is switched on or off. */
  onActive?: (on: boolean) => void;
}) {
  const { devices, refresh: refreshDevices } = useCameraDevices();
  const [deviceId, setDeviceId] = useState("");
  const [on, setOn] = useState(false);
  const [status, setStatus] = useState("");
  const [view, setView] = useState<{ n: number | null; g: number | null; stable: boolean }>({ n: null, g: null, stable: false });
  const videoRef = useRef<HTMLVideoElement>(null);
  const socketRef = useRef<WebSocket | null>(null);

  const onActiveRef = useRef(onActive);
  useEffect(() => { onActiveRef.current = onActive; });
  useEffect(() => {
    onActiveRef.current?.(on);
    return () => { if (on) onActiveRef.current?.(false); };
  }, [on]);

  useEffect(() => {
    if (!on) return;
    let cancelled = false;
    let raf = 0;
    let stream: MediaStream | null = null;
    let ws: WebSocket | null = null;

    (async () => {
      try {
        setStatus("Starting camera…");
        stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints(deviceId, 1280, 720), audio: false });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play();
        refreshDevices(); // labels only appear once permission is granted

        setStatus("Connecting to the gauge server…");
        ws = new WebSocket(GAUGE_URL);
        socketRef.current = ws;
        ws.binaryType = "arraybuffer";
        const grab = document.createElement("canvas");
        let busy = false;

        ws.onerror = () => {
          if (!cancelled) setStatus("No gauge server on 127.0.0.1:8766. Start it with: python -m gauge_reader.server");
        };
        ws.onclose = () => {
          force.current = null;
          if (!cancelled) setStatus((s) => (s.startsWith("No gauge server") ? s : "Gauge server disconnected"));
        };
        ws.onopen = () => { if (!cancelled) setStatus("Point the camera at the scale"); };
        ws.onmessage = (event) => {
          busy = false;
          if (cancelled) return;
          const m = JSON.parse(event.data as string) as {
            found?: boolean; newtons?: number | null; grams?: number | null; stable?: boolean;
            reason?: string; error?: string | null; tare?: boolean;
          };
          if ("tare" in m) { setStatus(m.tare ? "Zeroed" : `Tare failed: ${m.error ?? "unknown"}`); return; }
          if (m.error) { setStatus(`Gauge: ${m.error}`); return; }
          if (m.newtons == null || m.grams == null) {
            force.current = null;
            setView({ n: null, g: null, stable: false });
            setStatus(m.reason === "no-marker" || !m.reason ? "No marker found" : `No reading (${m.reason})`);
            return;
          }
          force.current = { n: m.newtons, g: m.grams, stable: !!m.stable, at: performance.now() };
          setView({ n: m.newtons, g: m.grams, stable: !!m.stable });
          setStatus(m.stable ? "Reading" : "Reading (settling)");
        };

        // One frame in flight at a time, so sending is paced by the server rather than queueing up.
        const loop = () => {
          raf = requestAnimationFrame(loop);
          if (busy || ws!.readyState !== WebSocket.OPEN || video.readyState < 2) return;
          busy = true;
          grab.width = video.videoWidth;
          grab.height = video.videoHeight;
          grab.getContext("2d")?.drawImage(video, 0, 0);
          grab.toBlob((blob) => {
            if (!blob || cancelled || ws!.readyState !== WebSocket.OPEN) { busy = false; return; }
            blob.arrayBuffer().then((buf) => {
              if (cancelled || ws!.readyState !== WebSocket.OPEN) { busy = false; return; }
              ws!.send(buf);
            });
          }, "image/jpeg", JPEG_QUALITY);
        };
        raf = requestAnimationFrame(loop);
      } catch (e) {
        console.error(e);
        if (!cancelled) {
          setStatus(`Force camera failed: ${e instanceof Error ? e.message : String(e)}`);
          setOn(false);
        }
      }
    })();

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
      ws?.close();
      socketRef.current = null;
      force.current = null;
      setView({ n: null, g: null, stable: false });
    };
  }, [on, deviceId, force, refreshDevices]);

  const shown = on || /^(Force camera failed|No gauge server)/.test(status) ? status : "";

  return (
    // Same three-row subgrid as HandTracker, so the two videos share a row, width and height.
    <div className="row-span-3 grid grid-rows-subgrid gap-y-2">
      <div className="flex flex-wrap content-start items-center gap-2">
        <CameraSelect label="Force camera" value={deviceId} devices={devices} onChange={setDeviceId} />
        <button className={on ? "btn btn-danger" : "btn"} onClick={() => setOn(!on)}>
          {on ? "Stop force" : "Start force"}
        </button>
        {on && (
          <button className="btn" onClick={() => socketRef.current?.readyState === WebSocket.OPEN && socketRef.current.send("tare")}>
            Zero (unloaded)
          </button>
        )}
        {shown && <span className="t-small muted">{shown}</span>}
      </div>
      <div>
        {on && (
          // 16:9 camera letterboxed into the shared 4:3 box, so the whole scale stays visible.
          <div className="relative aspect-[4/3] w-full overflow-hidden rounded bg-black">
            <video ref={videoRef} muted playsInline className="block h-full w-full object-contain" />
          </div>
        )}
      </div>
      <div className="space-y-2">
        {on && (
        <>
          <div className="flex items-center gap-2 t-body">
            <span>Force</span>
            <span className="num font-semibold">{view.n === null ? "–" : view.n.toFixed(2)} N</span>
            {view.g !== null && <span className="t-small muted">({view.g.toFixed(0)} gf)</span>}
          </div>
          <p className="t-small muted">
            Frames go to the gauge server on this machine, which measures them and discards them: nothing is written to disk.
            With no marker in view there is no force reading; the last value is never held.
          </p>
        </>
        )}
        {!on && !shown && (
          <p className="t-small faint">Pick the camera that sees the PESOLA scale, then Start force. Run the gauge server first.</p>
        )}
      </div>
    </div>
  );
}
