"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import MyoViewerClient from "./MyoViewerClient";
import HandTracker, { type HandPose } from "./HandTracker";
import GaugeForce, { freshGauge, type GaugeSample } from "./GaugeForce";
import { CORNERS, FACES, TURNS } from "../lib/viewPresets";
import { Bar, FINGER_COLOR, Legend, LineChart, type Band, type Series } from "./charts";
import ChatPanel from "./ChatPanel";
import GuideHand from "./GuideHand";
import PatientCard from "./PatientCard";
import { reportPage } from "../lib/markdownHtml";
import HandMeasurements from "./HandMeasurements";
import ViewCube from "./ViewCube";
import type { ViewApi } from "./MyoViewer";
import type { HandFrame } from "../lib/handFrame";
import MuscleCallout from "./MuscleCallout";
import DemoBadge from "./DemoBadge";
import SimulatedGauge from "./SimulatedGauge";
import { DEMO } from "../lib/demo";
import { onEmergencyStop } from "../lib/estop";
import type { MuscleAnchor } from "./MyoViewer";
import { buildContext, MUSCLE_NAMES, muscleInfo, resolveMuscle, type SessionContext } from "../lib/chatContext";
import { median } from "../lib/curlsim/dsp";
import { curveAt } from "../lib/curlsim/dsp";
import { genTrial, refOf } from "../lib/curlsim/generator";
import { analyzeTrial } from "../lib/curlsim/metrics";
import { LiveAnalyzer, medOfAnalysis, type LiveSnapshot } from "../lib/curlsim/live";
import {
  DEFAULT_WEIGHTS, GROUPS, metricsCsv, reportMd, reweight, runStudy, summaryCsv,
  type Fit, type Weights,
} from "../lib/curlsim/study";
import { medicalReportMd } from "../lib/curlsim/report";
import { FINGERS, FLEXOR_MUSCLES, SESSIONS, type Finger, type Trial } from "../lib/curlsim/types";

const pct = (v: number | null | undefined, d = 0) =>
  v == null || !Number.isFinite(v) ? "–" : `${(v * 100).toFixed(d)}%`;
const t90 = (f: Fit | null | undefined) =>
  !f ? "–" : Number.isFinite(f.t90) ? `${f.t90.toFixed(1)} wk` : "not reached";

function download(name: string, text: string, type: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

// The simulated index curl's controls (Run, speed, New patient) are hidden for now: the Rep machine is the
// way to run an exercise. Flip this to bring them back, nothing else about them was removed.
const SHOW_SIM_RUN_CONTROLS = false;

const FocusCtx = createContext<string | null>(null);

function Section({ id, title, children, note }: { id: string; title: string; children: ReactNode; note?: string }) {
  const focus = useContext(FocusCtx);
  return (
    <section id={`sec-${id}`} data-focus={focus === id ? "true" : undefined} className="section space-y-2 border-t border-(--line) pt-3">
      <h3 className="t-section">{title}</h3>
      {children}
      {note && <p className="t-small faint">{note}</p>}
    </section>
  );
}

const LIVE_ROWS: { key: keyof LiveSnapshot["current"] & string; label: string; unit: string; d: number }[] = [
  { key: "rom", label: "ROM so far", unit: "mm", d: 1 },
  { key: "peak_force", label: "Peak force so far", unit: "N", d: 1 },
  { key: "peak_velocity", label: "Peak velocity", unit: "mm/s", d: 0 },
  { key: "curl_time", label: "Curl time", unit: "s", d: 2 },
  { key: "time_to_peak", label: "Time to peak force", unit: "s", d: 2 },
  { key: "rfd_100", label: "RFD 100 ms", unit: "N/s", d: 0 },
  { key: "rfd_200", label: "RFD 200 ms", unit: "N/s", d: 0 },
  { key: "stiffness", label: "Apparent stiffness", unit: "N/mm", d: 2 },
  { key: "work_loading", label: "Loading work", unit: "mJ", d: 0 },
  { key: "f25", label: "F at 25% ref ROM", unit: "N", d: 1 },
  { key: "f50", label: "F at 50% ref ROM", unit: "N", d: 1 },
  { key: "f75", label: "F at 75% ref ROM", unit: "N", d: 1 },
  { key: "sparc", label: "Smoothness (SPARC)", unit: "", d: 2 },
  { key: "steadiness_cv", label: "Hold steadiness CV", unit: "%", d: 1 },
  { key: "force_relaxation_tau", label: "Relaxation tau", unit: "s", d: 1 },
  { key: "hysteresis", label: "Hysteresis", unit: "%", d: 0 },
];
/**
 * What the force channel is, in words, for everything a person or the assistant reads. In the
 * demo build the gauge panel is a simulation writing into the gauge's ref, so a reading that
 * arrived "from the gauge" must not be called a measurement anywhere.
 */
const GAUGE_SOURCE_NOTE = DEMO ? "simulated force, no gauge attached" : "measured force from the gauge";
const GAUGE_SOURCE_FIELD = DEMO ? "simulated (no gauge attached)" : "gauge reader (measured)";

const PHASE_COLOR: Record<string, string> = {
  baseline: "var(--phase-rest)", rest: "var(--phase-rest)", flexion: "var(--phase-flexion)",
  hold: "var(--phase-hold)", extension: "var(--phase-extension)",
};

function LivePanel({ snap, running }: { snap: LiveSnapshot; running: boolean }) {
  const inRep = snap.phase === "flexion" || snap.phase === "hold" || snap.phase === "extension";
  const last = snap.reps[snap.reps.length - 1];
  return (
    <Section id="live" title="Live analysis · index curl"
      note="Streaming analyzer: causal 10 Hz filter (adds ~0.1 s), phase detected on the fly. White = final for this rep, amber = still changing, dash = not computable yet. Hysteresis needs the full rep, CVs need 2+ reps, and the recovery index updates after each rep against the contralateral reference.">
      <div className="flex items-center gap-2">
        <span className="tag capitalize" style={{ background: PHASE_COLOR[snap.phase] }}>{snap.phase}</span>
        <span className="muted">{inRep ? `rep ${snap.repsDone + 1}` : "between reps"} · {snap.repsDone} done</span>
        <span className="ml-auto num muted">t = {snap.t.toFixed(1)} s {running ? "●" : "(finished)"}</span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
        {LIVE_ROWS.map((r) => {
          const v = inRep ? snap.current[r.key] : undefined;
          const shown = v !== undefined && Number.isFinite(v);
          const fin = snap.final.has(r.key);
          return (
            <div key={r.key} className="kv">
              <span>{r.label}</span>
              <span className={`num ${!shown ? "faint" : fin ? "text-(--ink)" : "text-(--warn)"}`}>
                {shown ? `${(v as number).toFixed(r.d)} ${r.unit}` : "–"}
              </span>
            </div>
          );
        })}
      </div>
      {snap.reps.length > 0 && (
        <table className="tbl mt-2">
          <thead><tr><th>Rep</th><th>ROM mm</th><th>Curl s</th><th>Peak N</th><th>RFD200</th><th>Hyst %</th></tr></thead>
          <tbody>
            {snap.reps.map((r) => (
              <tr key={r.rep}>
                <td>{r.rep}</td><td>{r.rom.toFixed(1)}</td><td>{r.curl_time.toFixed(2)}</td>
                <td>{r.peak_force.toFixed(1)}</td><td>{r.rfd_200.toFixed(0)}</td><td>{r.hysteresis.toFixed(0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {last && (
        <div className="mt-2 space-y-1.5">
          <div className="grid grid-cols-3 gap-2 t-small muted">
            <span>ROM CV {Number.isFinite(snap.trial.rom_cv) ? `${snap.trial.rom_cv.toFixed(1)}%` : "–"}</span>
            <span>Curl-time CV {Number.isFinite(snap.trial.curl_time_cv) ? `${snap.trial.curl_time_cv.toFixed(1)}%` : "–"}</span>
            <span>Force decay {Number.isFinite(snap.trial.force_decay) ? `${snap.trial.force_decay.toFixed(1)}%` : "–"}</span>
          </div>
          {snap.groups && (
            <>
              <div className="flex items-baseline justify-between">
                <span className="muted">Running recovery index (vs contralateral)</span>
                <span className="text-lg font-semibold" style={{ color: FINGER_COLOR.index }}>{pct(snap.recovery_index)}</span>
              </div>
              {GROUPS.map((g) => (
                <div key={g} className="grid grid-cols-[84px_1fr_36px] items-center gap-2">
                  <span className="capitalize muted">{g}</span>
                  <Bar value={snap.groups![g]} color={FINGER_COLOR.index} />
                  <span className="num text-right">{pct(snap.groups![g])}</span>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </Section>
  );
}

// `drops` are [t0, t1] spans (s) where the gauge was on but gave no fresh reading (recorded as 0 N);
// `open` is the start of the span in progress; `measured` is whether any sample used the gauge.
type CamBuf = { t: number[]; x: number[]; F: number[]; drops: [number, number][]; open: number | null; measured: boolean };
const newCamBuf = (): CamBuf => ({ t: [], x: [], F: [], drops: [], open: null, measured: false });

export default function SimApp() {
  const [seed, setSeed] = useState(1);
  const base = useMemo(() => runStudy(seed), [seed]);
  const [weights, setWeights] = useState<Weights>(DEFAULT_WEIGHTS);
  const study = useMemo(() => reweight(base, weights), [base, weights]);
  const [wi, setWi] = useState(SESSIONS.length - 1);
  const week = SESSIONS[wi];
  const [sel, setSel] = useState<Finger>("index");
  const [speed, setSpeed] = useState(1);
  const [live, setLive] = useState<{ finger: Finger; t: number } | null>(null);
  const [snap, setSnap] = useState<LiveSnapshot | null>(null);
  const [report, setReport] = useState<string | null>(null);

  const curlRef = useRef<{ joint: string; pip: string; flex: string[]; ext: string[]; target: number } | null>(null);
  const healthRef = useRef<Record<string, number>>({});
  const forceRef = useRef<{ n: number } | null>(null);
  const handPoseRef = useRef<HandPose | null>(null);
  const runToken = useRef(0);
  const camDispRef = useRef<number | null>(null);
  // Measured force from the gauge reader (null when absent or stale). `gaugeOn` is whether the user has
  // switched force sensing on; when it is off the old constant-20-N assumption still applies.
  const gaugeRef = useRef<GaugeSample | null>(null);
  const gaugeOn = useRef(false);
  const onGaugeActive = useCallback((on: boolean) => { gaugeOn.current = on; }, []);
  // The rep machine's simulated force, which in the demo build is what the force panel reports.
  // A ref, not state: it changes every frame and nothing up here needs to re-render for it.
  const repForceRef = useRef<number | null>(null);
  const onRepForce = useCallback((n: number | null) => { repForceRef.current = n; }, []);
  const camToken = useRef(0);
  const camAnalyzer = useRef<LiveAnalyzer | null>(null);
  const camBuf = useRef<CamBuf>(newCamBuf());
  const [camForce, setCamForce] = useState<{ measured: boolean; drops: [number, number][] }>({ measured: false, drops: [] });
  const [camTrial, setCamTrial] = useState<Trial | null>(null);
  const [camOn, setCamOn] = useState(false);

  // Chat tool-call targets: highlight set + screen anchors for floating callouts.
  const highlightRef = useRef<{ muscles: Set<string>; color: number } | null>(null);
  const anchorsRef = useRef<{ wanted: Set<string>; pos: Record<string, MuscleAnchor> }>({ wanted: new Set(), pos: {} });
  const [callouts, setCallouts] = useState<string[]>([]);
  const updateCallouts = useCallback((next: string[]) => {
    setCallouts(next);
    anchorsRef.current.wanted = new Set(next);
  }, []);
  const closeCallout = useCallback((m: string) => {
    setCallouts((cur) => {
      const next = cur.filter((x) => x !== m);
      anchorsRef.current.wanted = new Set(next);
      if (highlightRef.current) {
        const s = new Set(highlightRef.current.muscles);
        s.delete(m);
        highlightRef.current = s.size ? { ...highlightRef.current, muscles: s } : null;
      }
      return next;
    });
  }, []);

  // Explore mode: the mouse wheel steps through every tendon, highlighting it and opening its popup.
  const [explore, setExplore] = useState(false);
  const [exploreIdx, setExploreIdx] = useState(0);
  const exploreRef = useRef(false);
  const exploreIdxRef = useRef(0);
  const viewerWrapRef = useRef<HTMLDivElement>(null);
  // `focusMs`: also swing the camera round to face the muscle over that long (chat tour only; stepping
  // with the mouse wheel leaves the camera where the user put it).
  const showExplore = useCallback((i: number, focusMs?: number) => {
    const n = MUSCLE_NAMES.length;
    const idx = ((i % n) + n) % n;
    const m = MUSCLE_NAMES[idx];
    if (focusMs) viewApiRef.current?.focusMuscle(m, focusMs);
    exploreIdxRef.current = idx;
    setExploreIdx(idx);
    highlightRef.current = { muscles: new Set([m]), color: 0xfde047 };
    updateCallouts([m]);
  }, [updateCallouts]);
  const startExplore = useCallback(() => {
    exploreRef.current = true;
    setExplore(true);
    showExplore(0);
  }, [showExplore]);
  const tourTimer = useRef<ReturnType<typeof setInterval>>(undefined);
  const stopTour = useCallback(() => { clearInterval(tourTimer.current); tourTimer.current = undefined; }, []);
  useEffect(() => stopTour, [stopTour]);
  const stopExplore = useCallback(() => {
    stopTour();
    if (!exploreRef.current) return;
    exploreRef.current = false;
    setExplore(false);
    highlightRef.current = null;
    updateCallouts([]);
  }, [updateCallouts, stopTour]);
  // Tool call: auto-advance through every muscle popup in explore mode.
  const startTour = useCallback((sec: number) => {
    stopTour();
    exploreRef.current = true;
    setExplore(true);
    let i = 0;
    // The camera move takes most of the dwell time, but never longer than 1.2 s, so it has settled before the next muscle.
    const moveMs = Math.min(1200, sec * 600);
    showExplore(0, moveMs);
    tourTimer.current = setInterval(() => {
      i += 1;
      if (i >= MUSCLE_NAMES.length) { stopExplore(); return; }
      showExplore(i, moveMs);
    }, sec * 1000);
  }, [showExplore, stopExplore, stopTour]);
  useEffect(() => {
    const el = viewerWrapRef.current;
    if (!el) return;
    let acc = 0;
    let last = 0;
    const onWheel = (e: WheelEvent) => {
      if (!exploreRef.current) return;
      stopTour();
      e.preventDefault();
      e.stopPropagation();
      acc += e.deltaY;
      const now = performance.now();
      if (Math.abs(acc) < 40 || now - last < 120) return;
      showExplore(exploreIdxRef.current + (acc > 0 ? 1 : -1));
      acc = 0;
      last = now;
    };
    el.addEventListener("wheel", onWheel, { passive: false, capture: true });
    return () => el.removeEventListener("wheel", onWheel, { capture: true });
  }, [showExplore, stopTour]);
  useEffect(() => {
    if (!explore) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") stopExplore(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [explore, stopExplore]);

  const [focusSec, setFocusSec] = useState<string | null>(null);
  const focusTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(focusTimer.current), []);
  const onTool = useCallback((name: string, args: Record<string, unknown>) => {
    const resolve = (v: unknown) => (typeof v === "string" ? resolveMuscle(v) : null);
    const select = (m: string) => { const f = muscleInfo(m)?.finger; if (f) setSel(f); };
    if (name === "tour_muscles") {
      const raw = Number(args.seconds_each);
      const sec = Number.isFinite(raw) ? Math.min(10, Math.max(1, raw)) : 3;
      startTour(sec);
      return `touring all ${MUSCLE_NAMES.length} muscle popups, ${sec}s each (about ${Math.round(MUSCLE_NAMES.length * sec)}s total); it stops on its own at the end`;
    }
    stopTour();
    if (name !== "show_dashboard_section" && exploreRef.current) { exploreRef.current = false; setExplore(false); }
    if (name === "show_dashboard_section") {
      const id = String(args.section ?? "");
      const el = document.getElementById(`sec-${id}`);
      if (!el) return `error: section "${id}" is not on screen${id === "live" ? " (run the index curl first)" : ""}`;
      el.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
      setFocusSec(id);
      clearTimeout(focusTimer.current);
      focusTimer.current = setTimeout(() => setFocusSec(null), 6000);
      return `scrolled to and highlighted the "${id}" section of the dashboard`;
    }
    if (name === "set_camera_view" || name === "rotate_camera") {
      const view = viewApiRef.current;
      if (!view) return "error: the 3D view is not ready yet";
      if (name === "rotate_camera") {
        const turn = TURNS[String(args.direction) as keyof typeof TURNS];
        if (!turn) return `error: unknown direction ${JSON.stringify(args.direction)}; use ${Object.keys(TURNS).join(", ")}`;
        const raw = Number(args.degrees);
        const deg = Number.isFinite(raw) ? Math.min(360, Math.max(1, raw)) : 90;
        view.turn(turn[0], turn[1] * (deg * Math.PI) / 180);
        return `rotating the camera ${args.direction} by ${deg} degrees`;
      }
      const id = String(args.view ?? "");
      if (id === "home") { view.home(); return "returning the camera to the starting side profile"; }
      const preset = FACES.find((f) => f.id === id) ?? CORNERS.find((c) => c.id === id);
      if (!preset) return `error: unknown view ${JSON.stringify(args.view)}`;
      view.snapTo(preset.dir, preset.up);
      return `rotating the camera to the ${"label" in preset ? preset.label : `${preset.name} corner (three-quarter)`} view`;
    }
    if (name === "clear_highlights") {
      setFocusSec(null);
      highlightRef.current = null;
      updateCallouts([]);
      return "cleared";
    }
    if (name === "highlight_muscles") {
      const list = Array.isArray(args.muscles) ? args.muscles : [];
      const ok = list.map(resolve).filter((x): x is string => !!x);
      const bad = list.filter((v) => !resolve(v));
      if (!ok.length) return `error: unknown muscles ${JSON.stringify(bad)}`;
      highlightRef.current = { muscles: new Set(ok), color: 0xfde047 };
      updateCallouts(ok.slice(0, 4)); // every highlighted muscle gets a pointer box
      viewApiRef.current?.focusMuscle(ok[0]);
      select(ok[0]);
      return `highlighted ${ok.join(", ")}${bad.length ? `; unknown: ${bad.join(", ")}` : ""}`;
    }
    if (name === "show_muscle_info") {
      const m = resolve(args.muscle);
      if (!m) return `error: unknown muscle ${JSON.stringify(args.muscle)}`;
      const cur = highlightRef.current?.muscles ?? new Set<string>();
      highlightRef.current = { muscles: new Set([...cur, m]), color: 0xfde047 };
      setCallouts((c) => {
        const next = c.includes(m) ? c : [...c, m].slice(-4);
        anchorsRef.current.wanted = new Set(next);
        return next;
      });
      viewApiRef.current?.focusMuscle(m);
      select(m);
      return `showing floating info box for ${m}`;
    }
    return `error: unknown tool ${name}`;
  }, [updateCallouts, startTour, stopTour]);

  // Tint flexor tendons in the 3D model by the *estimated* recovery of each unit.
  useEffect(() => {
    const h: Record<string, number> = {};
    for (const st of study.structures.filter((s) => s.session === week)) {
      const m = FLEXOR_MUSCLES[st.finger];
      if (m.fdp) h[m.fdp] = st.fdp;
      if (m.fds && st.fds != null) h[m.fds] = st.fds;
    }
    healthRef.current = h;
  }, [study, week]);

  const stop = useCallback(() => {
    runToken.current++;
    curlRef.current = null;
    forceRef.current = null;
    setLive(null);
  }, []);
  useEffect(() => stop, [stop]);

  const CAM_FS = 100;
  const publishCamTrial = useCallback(() => {
    const b = camBuf.current;
    if (b.t.length < 60) return;
    const tEnd = b.t[b.t.length - 1];
    setCamForce({ measured: b.measured, drops: b.open === null ? [...b.drops] : [...b.drops, [b.open, tEnd]] });
    setCamTrial({
      patientId: base.patient.id, finger: "index", hand: "injured", session: week, trialIndex: 1, test: "full",
      fs: CAM_FS, t: Float64Array.from(b.t), x: Float64Array.from(b.x), F: Float64Array.from(b.F),
    });
  }, [base.patient.id, week]);

  const stopCam = useCallback(() => {
    camToken.current++;
    if (camAnalyzer.current) { setSnap(camAnalyzer.current.snapshot()); publishCamTrial(); }
    camAnalyzer.current = null;
    forceRef.current = null;
    setCamOn(false);
  }, [publishCamTrial]);
  useEffect(() => () => { camToken.current++; }, []);

  // Camera mode: the tracked index-tip displacement drives the live analysis in place of the simulated
  // curl. Force comes from the gauge reader when force sensing is on (a dropout is recorded as 0 N and
  // marked on the chart); otherwise it is assumed constant at 20 N with +/-1 N uniform noise. The analyzer
  // zeroes its baseline over the first 0.2 s, so the assumed force is held at 0 during that tare window.
  const startCam = useCallback(() => {
    const token = ++camToken.current;
    const FS = CAM_FS;
    const conA = analyzeTrial(genTrial(base.patient, "index", "contralateral", week, 1, "full"));
    const refRomLive = median(conA.reps.map((r) => r.rom));
    const analyzer = new LiveAnalyzer(FS, refRomLive, conA.reps.length ? medOfAnalysis(conA) : null, weights);
    const tare = Math.round(0.2 * FS);
    camAnalyzer.current = analyzer;
    camBuf.current = newCamBuf();
    setCamForce({ measured: false, drops: [] });
    setCamTrial(null);
    setSel("index");
    setSnap(null);
    setReport(null);
    setCamOn(true);
    let start: number | null = null;
    let pushed = 0;
    let lastSnap = 0;
    const tick = (now: number) => {
      if (camToken.current !== token) return;
      const d = camDispRef.current;
      if (d !== null) {
        if (start === null) start = now;
        const n = Math.floor(((now - start) / 1000) * FS);
        const b = camBuf.current;
        for (; pushed <= n; pushed++) {
          // Measured force when the gauge is on; a dropout is 0 N, never the last value. Otherwise the
          // constant-force assumption (zero during the tare window).
          const g = gaugeOn.current ? freshGauge(gaugeRef) : null;
          const f = gaugeOn.current ? (g ? g.n : 0) : pushed < tare ? 0 : 20 + (Math.random() * 2 - 1);
          analyzer.push(pushed / FS, d, f);
          if (b.t.length < 30000) {
            const ts = pushed / FS;
            b.t.push(ts); b.x.push(d); b.F.push(f);
            if (gaugeOn.current) {
              b.measured = true;
              if (!g && b.open === null) b.open = ts;
              else if (g && b.open !== null) { b.drops.push([b.open, ts]); b.open = null; }
            } else if (b.open !== null) { b.drops.push([b.open, ts]); b.open = null; }
          }
        }
        if (now - lastSnap > 100) { lastSnap = now; setSnap(analyzer.snapshot()); publishCamTrial(); }
      }
      // The fingertip arrow follows the measured force; absent or stale means no arrow.
      const measured = gaugeOn.current ? freshGauge(gaugeRef) : null;
      forceRef.current = measured ? { n: measured.n } : null;
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [base.patient, week, weights, publishCamTrial]);

  const onCamActive = useCallback((on: boolean) => { if (on) startCam(); else stopCam(); }, [startCam, stopCam]);

  // The tracker's on/off state lives here so the rep machine panel can start the camera too.
  // Starting it from outside must not leave the simulated curl running underneath it, which
  // pressing the tracker's own button already prevents through its `onStart={stop}`.
  const [trackerOn, setTrackerOn] = useState(false);
  const requestTracker = useCallback((on: boolean) => {
    if (on) stop();
    setTrackerOn(on);
  }, [stop]);

  // The stop in the top bar covers everything this page drives, not only the rep machine: the
  // simulated curl stops and the camera is handed back. Switching the tracker off takes the
  // camera-mode analysis down with it, through HandTracker's own onActive.
  useEffect(() => onEmergencyStop(() => { stop(); setTrackerOn(false); }), [stop]);

  // The main camera's API, handed up by the viewer for the view cube in the top right.
  const viewApiRef = useRef<ViewApi | null>(null);
  const onViewApi = useCallback((api: ViewApi | null) => { viewApiRef.current = api; }, []);

  // One HandFrame stream from the tracker, fanned out to whatever panels want every frame.
  const handFrameSubs = useRef(new Set<(frame: HandFrame) => void>());
  const onHandFrame = useCallback((frame: HandFrame) => { handFrameSubs.current.forEach((fn) => fn(frame)); }, []);
  const subscribeHandFrames = useCallback((fn: (frame: HandFrame) => void) => {
    handFrameSubs.current.add(fn);
    return () => { handFrameSubs.current.delete(fn); };
  }, []);

  const run = useCallback(() => {
    const token = ++runToken.current;
    // Playback is index-finger only, with a deliberately subtle curl.
    const trials = [genTrial(base.patient, "index", "injured", week, 1, "full")];
    // Live analysis reference: contralateral index finger, same session.
    const conA = analyzeTrial(genTrial(base.patient, "index", "contralateral", week, 1, "full"));
    const refRomLive = median(conA.reps.map((r) => r.rom));
    const analyzer = new LiveAnalyzer(trials[0].fs, refRomLive, conA.reps.length ? medOfAnalysis(conA) : null, weights);
    let pushed = 0;
    let lastSnap = 0;
    setSnap(null);
    setReport(null);
    setCamTrial(null);
    const MAX_MCP = 0.55; // rad (~31 deg) target for a full-ROM curl: deliberately subtle
    let fi = 0;
    let start = performance.now();
    let lastUi = 0;
    const tick = (now: number) => {
      if (runToken.current !== token) return;
      const tr = trials[fi];
      const tt = ((now - start) / 1000) * speed;
      if (tt >= tr.t[tr.t.length - 1]) {
        fi++;
        start = now;
        if (fi >= trials.length) { setSnap(analyzer.snapshot()); stop(); return; }
      } else {
        const i = Math.min(Math.floor(tt * tr.fs), tr.x.length - 1);
        const m = FLEXOR_MUSCLES[tr.finger];
        // Feed every new sample to the streaming analyzer (as a real-time device would).
        for (; pushed <= i; pushed++) analyzer.push(tr.t[pushed], tr.x[pushed], tr.F[pushed]);
        if (now - lastSnap > 100) { lastSnap = now; setSnap(analyzer.snapshot()); }
        // Smooth the (noisy) displacement, then map displacement -> MCP angle target.
        let acc = 0;
        let facc = 0;
        let cnt = 0;
        for (let k = Math.max(i - 5, 0); k <= Math.min(i + 5, tr.x.length - 1); k++) { acc += tr.x[k] - tr.x[0]; facc += tr.F[k] - tr.F[0]; cnt++; }
        // Fingertip force (N), smoothed and baseline-zeroed, drawn as an arrow at the fingertip.
        forceRef.current = { n: Math.max(facc / cnt, 0) };
        const u = Math.min(Math.max(acc / cnt / refOf(tr.finger).rom, 0), 1);
        curlRef.current = { joint: "mcp2_flexion_r", pip: "pm2_flexion_r", flex: [m.fds ?? "", m.fdp ?? ""].filter(Boolean), ext: m.ext, target: MAX_MCP * u };
        if (now - lastUi > 60) { lastUi = now; setSel(tr.finger); setLive({ finger: tr.finger, t: tt }); }
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [base.patient, week, speed, weights, stop]);

  const view = useMemo(() => {
    const con = genTrial(base.patient, sel, "contralateral", week, 1, "full");
    const inj = genTrial(base.patient, sel, "injured", week, 1, "full");
    const aC = analyzeTrial(con);
    const refRom = median(aC.reps.map((r) => r.rom));
    const rr = Number.isFinite(refRom) ? refRom : undefined;
    // Camera data overrides the simulated injured-hand trial for the index finger.
    let cam = false;
    let aI = analyzeTrial(inj, rr);
    if (sel === "index" && camTrial) {
      try { aI = analyzeTrial(camTrial, rr); cam = true; } catch { /* keep simulated */ }
    }
    return { aC, aI, cam };
  }, [base.patient, sel, week, camTrial]);
  const cursorT = camOn && view.cam && sel === "index"
    ? view.aI.t[view.aI.t.length - 1]
    : live?.finger === sel ? live.t : undefined;

  const bands: Band[] = view.aI.segs.flatMap((g) => {
    const t = view.aI.t;
    return [
      { x0: t[g.s], x1: t[g.flexEnd], color: "rgba(56,189,248,0.14)" },
      { x0: t[g.flexEnd], x1: t[g.extStart], color: "rgba(245,158,11,0.14)" },
      { x0: t[g.extStart], x1: t[g.e], color: "rgba(52,211,153,0.14)" },
    ];
  });
  // Force dropouts (gauge on, no reading) in red, on the camera trial only.
  const dropBands: Band[] = view.cam
    ? camForce.drops.map(([x0, x1]) => ({ x0, x1, color: "rgba(239,68,68,0.28)" }))
    : [];
  const dropSecs = camForce.drops.reduce((s, [a, b]) => s + (b - a), 0);
  const loop = (a: typeof view.aI, color: string, name: string): Series | null => {
    const g = a.segs[0];
    if (!g) return null;
    return { name, color, xs: Array.from(a.xf.subarray(g.s, g.e + 1)), ys: Array.from(a.Ff.subarray(g.s, g.e + 1)) };
  };
  const loopSeries = [loop(view.aC, "#9ca3af", "contralateral (reference)"), loop(view.aI, FINGER_COLOR[sel], `${sel} injured`)].filter(Boolean) as Series[];

  // What the chat sees of the measured curls: the session metrics and the medical report, both only once
  // a rep has been analysed. The report is the same text the Export button opens.
  const chatContext = useMemo(() => {
    const hasReps = !!snap && snap.reps.length > 0;
    const measured: SessionContext | null = snap && hasReps ? {
      source: camTrial ? "camera" : "simulated",
      inProgress: !!live || camOn,
      forceSource: !camTrial ? "simulation" : camForce.measured ? GAUGE_SOURCE_FIELD : "assumed constant 20 N",
      forceDropouts: camTrial && camForce.measured ? { count: camForce.drops.length, totalSeconds: dropSecs } : null,
      repsAnalysed: snap.reps.length,
      recoveryIndex: snap.recovery_index,
      groupScores: snap.groups,
      medianMetrics: snap.med,
      trialVariability: snap.trial,
      perRep: snap.reps,
    } : null;
    return buildContext(
      study, week, sel,
      snap ? { phase: snap.phase, repsDone: snap.repsDone, recovery_index: snap.recovery_index ?? undefined } : null,
      { measured, medicalReport: snap && hasReps ? (report ?? medicalReportMd(study, week, snap)) : null },
    );
  }, [study, week, sel, snap, camTrial, camForce, camOn, live, report, dropSecs]);

  const weeks = SESSIONS;
  const riSeries: Series[] = FINGERS.flatMap((f) => {
    const rows = study.summary.filter((r) => r.finger === f);
    const fit = study.fits[f];
    const grid = Array.from({ length: 49 }, (_, i) => i * 0.5);
    const s: Series[] = [{ name: f, color: FINGER_COLOR[f], xs: rows.map((r) => r.session), ys: rows.map((r) => r.recovery_index * 100), dots: true, width: 0 }];
    if (fit) s.push({ name: `${f} fit`, color: FINGER_COLOR[f], xs: grid, ys: grid.map((t) => curveAt(fit, t) * 100), width: 1.5 });
    return s;
  });

  const rowsNow = study.summary.filter((r) => r.session === week);
  const stNow = study.structures.filter((s) => s.session === week);
  const sens = study.sensitivity;
  const sumW = GROUPS.reduce((a, g) => a + weights[g], 0) || 1;

  return (
    <FocusCtx.Provider value={focusSec}>
    <div className="relative h-full w-full">
      <div ref={viewerWrapRef} className="absolute inset-0">
        <MyoViewerClient curl={curlRef} health={healthRef} highlight={highlightRef} anchors={anchorsRef} force={forceRef} handPose={handPoseRef} explore={exploreRef}
          exploreOn={explore}
          exploreLabel={`${exploreIdx + 1}/${MUSCLE_NAMES.length} ${muscleInfo(MUSCLE_NAMES[exploreIdx])?.label ?? ""}`}
          onExploreToggle={explore ? stopExplore : startExplore}
          onViewApi={onViewApi} />
      <ViewCube api={viewApiRef} />
      </div>
      <MuscleCallout callouts={callouts} anchors={anchorsRef} study={study} week={week} onClose={closeCallout} />
      {/* The left column: the rep machine with the chat under it, stacked from the bottom up so
          that collapsing the chat lowers the rep machine instead of leaving a hole, and a shorter
          screen loses the least important thing last. The column is transparent to the mouse so
          the 3D view behind it still drags and the explore wheel still works. */}
      <div className="pointer-events-none absolute bottom-4 left-4 top-4 flex w-[340px] max-w-[calc(100%-2rem)] flex-col justify-end gap-4">
        <GuideHand subscribe={subscribeHandFrames} trackerOn={trackerOn} onTrackerChange={requestTracker} onForce={onRepForce} />
        <ChatPanel context={chatContext} onTool={onTool} />
      </div>
      <aside className="panel absolute bottom-4 right-4 top-4 w-[470px] max-w-[calc(100%-2rem)] space-y-4 overflow-y-auto">
        <PatientCard
          onExport={snap && snap.reps.length > 0 && !live && !camOn ? () => {
            const md = report ?? medicalReportMd(study, week, snap);
            if (!report) setReport(md);
            const url = URL.createObjectURL(new Blob([reportPage(`Medical report ${study.patient.id} week ${week}`, md)], { type: "text/html" }));
            window.open(url, "_blank");
            setTimeout(() => URL.revokeObjectURL(url), 60_000);
          } : undefined}
        />
        <div>
          <div className="flex items-center gap-2">
            <span className="t-title">Flexor recovery simulator</span>
            <DemoBadge kind="simulated" label="Synthetic patient" />
          </div>
          <p className="t-small muted mt-1">
            Synthetic patient {study.patient.id}: the 3D hand curls the index finger (subtle flexion); every finger is simulated, force and
            displacement are analysed against the contralateral hand, and the healing course of
            every flexor tendon unit is fitted. Tendons are tinted red→green by estimated recovery.
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-1">
            <span className="mr-1 muted">Week</span>
            {weeks.map((w, i) => (
              <button key={w} onClick={() => setWi(i)} aria-pressed={i === wi} className="chip">{w}</button>
            ))}
          </div>
          {SHOW_SIM_RUN_CONTROLS && (
            <div className="flex flex-wrap items-center gap-2">
              {live ? (
                <button onClick={stop} className="btn btn-danger">Stop</button>
              ) : (
                <button onClick={run} disabled={camOn} className="btn btn-primary">Run index curl</button>
              )}
              <select value={speed} onChange={(e) => setSpeed(+e.target.value)} className="field">
                {[1, 2, 4].map((s) => <option key={s} value={s} className="text-black">{s}× speed</option>)}
              </select>
              <button onClick={() => setSeed((s) => s + 1)} className="btn">New patient</button>
            </div>
          )}
          {/* Hand camera and force camera side by side; each column holds its own picker, status and preview. */}
          {/* Both columns are 3-row subgrids (controls, video, readout) of this grid, with a divider between. */}
          <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] grid-rows-[auto_auto_auto] gap-x-5 p-1.5">
            <HandTracker pose={handPoseRef} disp={camDispRef} onActive={onCamActive} onFrame={onHandFrame} on={trackerOn} onOnChange={setTrackerOn} />
            <div aria-hidden className="row-span-3 w-px self-stretch" style={{ background: "var(--line)" }} />
            {/* With no bench there is no gauge to connect to, so the demo build puts a labelled
                simulation in its place rather than a panel that can only fail to connect. */}
            {DEMO
              ? <SimulatedGauge forceRef={gaugeRef} onActive={onGaugeActive} driven={repForceRef} />
              : <GaugeForce force={gaugeRef} onActive={onGaugeActive} />}
          </div>
          <div className="flex gap-1">
            {FINGERS.map((f) => (
              <button key={f} onClick={() => setSel(f)} aria-pressed={sel === f}
                className="chip flex-1 capitalize"
                style={sel === f ? { background: FINGER_COLOR[f], color: "#0b0d11" } : undefined}>
                {f}{live?.finger === f ? " ●" : ""}
              </button>
            ))}
          </div>
        </div>

        <Section id="signals" title={`${sel} · ${view.cam ? "camera" : `week ${week}`} · displacement x(t) and force F(t)`}
          note={`Shaded: flexion (blue), hold (amber), extension (green). Filtered, force baseline zeroed over first 0.2 s.${dropBands.length ? " Red: force dropouts (no gauge reading), recorded as 0 N." : ""}`}>
          <LineChart h={110} yLabel="x (mm)" series={[{ name: "x", color: FINGER_COLOR[sel], xs: view.aI.t, ys: view.aI.xf }]}
            bands={bands} cursor={cursorT} />
          <LineChart h={110} yLabel="F (N)" xLabel="time (s)" series={[{ name: "F", color: "#e5e7eb", xs: view.aI.t, ys: view.aI.Ff }]}
            bands={[...bands, ...dropBands]} cursor={cursorT} />
          <div className="t-small muted">
            {view.aI.ok
              ? `${view.aI.reps.length} clean reps`
              : view.cam
                // The offline checker can reject a camera trial for force baseline drift: a constant 20 N
                // assumption reads as drift, and so can a gauge dropout (0 N). Neither means a bad trial.
                ? `${view.aI.reps.length} reps detected (camera, ${camForce.measured ? GAUGE_SOURCE_NOTE : "constant 20 N force assumed"})`
                : `Trial rejected: ${view.aI.reasons.join("; ")}`}
            {view.cam && camForce.drops.length > 0 &&
              ` · ${camForce.drops.length} force dropout${camForce.drops.length === 1 ? "" : "s"}, ${dropSecs.toFixed(1)} s total (0 N)`}
          </div>
        </Section>

        {snap && <LivePanel snap={snap} running={!!live || camOn} />}

        {camOn && <HandMeasurements subscribe={subscribeHandFrames} />}

        <Section id="recovery_table" title="Recovery index per finger" note="Recovery index = weighted mean of motion / strength / mechanics / consistency scores vs contralateral (capped 0–120 %). R∞, τ from fit R(t)=R∞(1−e^(−t/τ)); time-to-90 % is when the fitted index reaches 90 %.">
          <table className="tbl">
            <thead><tr><th>Finger</th><th>Index</th><th>R∞</th><th>τ (wk)</th><th>t90</th></tr></thead>
            <tbody>
              {FINGERS.map((f) => {
                const r = rowsNow.find((x) => x.finger === f);
                const fit = study.fits[f];
                return (
                  <tr key={f}>
                    <td className="capitalize" style={{ color: FINGER_COLOR[f] }}>{f}</td>
                    <td>{r ? pct(r.recovery_index) : "skipped"}</td>
                    <td>{pct(fit?.rinf)}</td>
                    <td>{fit ? fit.tau.toFixed(1) : "–"}</td>
                    <td>{t90(fit)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Section>

        <Section id="recovery_curve" title="Recovery index vs week">
          <LineChart h={170} yLabel="% of contralateral" xLabel="weeks since surgery" yDomain={[0, 120]} xDomain={[0, 24]} series={riSeries} />
          <Legend items={FINGERS.map((f) => ({ name: f, color: FINGER_COLOR[f] }))} />
        </Section>

        <Section id="group_scores" title={`Group scores · week ${week}`}>
          <div className="space-y-1.5">
            {FINGERS.map((f) => {
              const r = rowsNow.find((x) => x.finger === f);
              return (
                <div key={f} className="grid grid-cols-[52px_1fr_1fr_1fr_1fr] items-center gap-2">
                  <span className="capitalize" style={{ color: FINGER_COLOR[f] }}>{f}</span>
                  {GROUPS.map((g) => <Bar key={g} value={r?.groups[g] ?? NaN} color={FINGER_COLOR[f]} />)}
                </div>
              );
            })}
            <div className="grid grid-cols-[52px_1fr_1fr_1fr_1fr] gap-2 t-small faint">
              <span />{GROUPS.map((g) => <span key={g}>{g}</span>)}
            </div>
          </div>
        </Section>

        <Section id="force_loop" title={`Force–displacement loop · ${sel} · week ${week}`} note="Apparent/effective behaviour of the whole flexor system (tendon + muscle + adhesions + soft tissue). Not tendon stiffness.">
          <LineChart h={160} xLabel="x (mm)" yLabel="F (N)" series={loopSeries} />
          <Legend items={loopSeries.map((s) => ({ name: s.name, color: s.color }))} />
        </Section>

        <Section id="tendon_units" title={`Flexor tendon units · week ${week} (model-attributed)`}
          note="Attributed from peak force: FDS via a DIP-blocked curl, FDP/FPL = full curl − FDS. Fit columns are for the full week series. 'Sim truth' is the generator's hidden value, shown to validate the attribution. Ligaments and pulleys are not in the MyoHand model, so they are not characterized.">
          <table className="tbl">
            <thead><tr><th>Unit</th><th>Est.</th><th>Sim truth</th><th>R∞</th><th>τ</th><th>t90</th></tr></thead>
            <tbody>
              {stNow.flatMap((s) => {
                const rows: { key: string; label: string; est: number; truth: number; fit: Fit | null }[] = [
                  { key: `${s.finger}:fdp`, label: s.finger === "thumb" ? "FPL" : `FDP ${s.finger}`, est: s.fdp, truth: s.trueFdp, fit: study.structureFits[`${s.finger}:fdp`] },
                ];
                if (s.fds != null && s.trueFds != null)
                  rows.push({ key: `${s.finger}:fds`, label: `FDS ${s.finger}`, est: s.fds, truth: s.trueFds, fit: study.structureFits[`${s.finger}:fds`] });
                return rows.map((r) => (
                  <tr key={r.key}>
                    <td style={{ color: FINGER_COLOR[s.finger] }}>{r.label}</td>
                    <td>{pct(r.est)}</td><td className="muted">{pct(Math.min(r.truth, 1.2))}</td>
                    <td>{pct(r.fit?.rinf)}</td><td>{r.fit ? r.fit.tau.toFixed(1) : "–"}</td><td>{t90(r.fit)}</td>
                  </tr>
                ));
              })}
            </tbody>
          </table>
        </Section>

        <Section id="weights" title="Weights and sensitivity" note="The index is a heuristic composite; the weights are subjective.">
          {GROUPS.map((g) => (
            <label key={g} className="grid grid-cols-[84px_1fr_36px] items-center gap-2">
              <span className="capitalize">{g}</span>
              <input type="range" min={0} max={1} step={0.05} value={weights[g]}
                onChange={(e) => setWeights({ ...weights, [g]: +e.target.value })} />
              <span className="num muted text-right">{((weights[g] / sumW) * 100).toFixed(0)}%</span>
            </label>
          ))}
          <button className="btn btn-sm" onClick={() => setWeights(DEFAULT_WEIGHTS)}>Reset weights</button>
          <table className="tbl">
            <thead><tr><th>Finger (wk {sens.session})</th><th>Base</th><th>Range at ±50 % weight</th></tr></thead>
            <tbody>
              {sens.baseRanking.map((f) => (
                <tr key={f}>
                  <td className="capitalize" style={{ color: FINGER_COLOR[f] }}>{f}</td>
                  <td>{pct(sens.perFinger[f].base)}</td>
                  <td>{pct(sens.perFinger[f].min)} – {pct(sens.perFinger[f].max)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="t-small muted">
            Ranking {sens.nChanged === 0 ? "never changed" : `changed in ${sens.nChanged}`} across {sens.scenarios.length} one-at-a-time weight scenarios.
          </div>
        </Section>

        <Section id="data_quality" title="Data quality">
          {study.flags.length === 0 ? <div className="muted">No flags.</div> : (
            <ul className="max-h-28 list-disc space-y-0.5 overflow-y-auto pl-4 t-small text-(--warn)">
              {study.flags.map((f, i) => <li key={i}>{f}</li>)}
            </ul>
          )}
        </Section>

        <Section id="export" title="Export">
          <div className="flex flex-wrap gap-2">
            <button className="btn" onClick={() => download("metrics.csv", metricsCsv(study), "text/csv")}>metrics.csv</button>
            <button className="btn" onClick={() => download("summary.csv", summaryCsv(study), "text/csv")}>summary.csv</button>
            <button className="btn" onClick={() => download(`REPORT_${study.patient.id}.md`, reportMd(study), "text/markdown")}>REPORT.md</button>
          </div>
        </Section>

        <p className="t-small faint border-t border-(--line) pt-3">
          Only flexor-driven curl behaviour is characterized; extensors, FDP vs FDS beyond the
          DIP-block attribution, thumb-specific tendons and intrinsics are not measured. All data
          here are simulated. Force–displacement properties are apparent, not true tendon
          stress/strain/stiffness. For research and demonstration, not clinical decisions.
        </p>
      </aside>
    </div>
    </FocusCtx.Provider>
  );
}
