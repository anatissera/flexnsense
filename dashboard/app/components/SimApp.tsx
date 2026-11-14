"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import MyoViewerClient from "./MyoViewerClient";
import { Bar, FINGER_COLOR, Legend, LineChart, type Band, type Series } from "./charts";
import ChatPanel from "./ChatPanel";
import MuscleCallout from "./MuscleCallout";
import type { MuscleAnchor } from "./MyoViewer";
import { buildContext, muscleInfo, resolveMuscle } from "../lib/chatContext";
import { median } from "../lib/curlsim/dsp";
import { curveAt } from "../lib/curlsim/dsp";
import { genTrial, refOf } from "../lib/curlsim/generator";
import { analyzeTrial } from "../lib/curlsim/metrics";
import { LiveAnalyzer, medOfAnalysis, type LiveSnapshot } from "../lib/curlsim/live";
import {
  DEFAULT_WEIGHTS, GROUPS, metricsCsv, reportMd, reweight, runStudy, summaryCsv,
  type Fit, type Weights,
} from "../lib/curlsim/study";
import { FINGERS, FLEXOR_MUSCLES, SESSIONS, type Finger } from "../lib/curlsim/types";

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

  const curlRef = useRef<{ joint: string; pip: string; flex: string[]; ext: string[]; target: number } | null>(null);
  const healthRef = useRef<Record<string, number>>({});
  const runToken = useRef(0);

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
  const [focusSec, setFocusSec] = useState<string | null>(null);
  const focusTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(focusTimer.current), []);
  const onTool = useCallback((name: string, args: Record<string, unknown>) => {
    const resolve = (v: unknown) => (typeof v === "string" ? resolveMuscle(v) : null);
    const select = (m: string) => { const f = muscleInfo(m)?.finger; if (f) setSel(f); };
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
      select(m);
      return `showing floating info box for ${m}`;
    }
    return `error: unknown tool ${name}`;
  }, [updateCallouts]);

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
    setLive(null);
  }, []);
  useEffect(() => stop, [stop]);

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
        let cnt = 0;
        for (let k = Math.max(i - 5, 0); k <= Math.min(i + 5, tr.x.length - 1); k++) { acc += tr.x[k] - tr.x[0]; cnt++; }
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
    const aI = analyzeTrial(inj, Number.isFinite(refRom) ? refRom : undefined);
    return { aC, aI };
  }, [base.patient, sel, week]);

  const bands: Band[] = view.aI.segs.flatMap((g) => {
    const t = view.aI.t;
    return [
      { x0: t[g.s], x1: t[g.flexEnd], color: "rgba(56,189,248,0.14)" },
      { x0: t[g.flexEnd], x1: t[g.extStart], color: "rgba(245,158,11,0.14)" },
      { x0: t[g.extStart], x1: t[g.e], color: "rgba(52,211,153,0.14)" },
    ];
  });
  const loop = (a: typeof view.aI, color: string, name: string): Series | null => {
    const g = a.segs[0];
    if (!g) return null;
    return { name, color, xs: Array.from(a.xf.subarray(g.s, g.e + 1)), ys: Array.from(a.Ff.subarray(g.s, g.e + 1)) };
  };
  const loopSeries = [loop(view.aC, "#9ca3af", "contralateral (reference)"), loop(view.aI, FINGER_COLOR[sel], `${sel} injured`)].filter(Boolean) as Series[];

  const chatContext = useMemo(
    () => buildContext(study, week, sel, snap ? { phase: snap.phase, repsDone: snap.repsDone, recovery_index: snap.recovery_index ?? undefined } : null),
    [study, week, sel, snap],
  );

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
      <MyoViewerClient curl={curlRef} health={healthRef} highlight={highlightRef} anchors={anchorsRef} />
      <MuscleCallout callouts={callouts} anchors={anchorsRef} study={study} week={week} onClose={closeCallout} />
      <ChatPanel context={chatContext} onTool={onTool} />
      <aside className="panel absolute bottom-4 right-4 top-4 w-[470px] max-w-[calc(100%-2rem)] space-y-4 overflow-y-auto">
        <div>
          <div className="t-title">Flexor recovery simulator</div>
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
          <div className="flex flex-wrap items-center gap-2">
            {live ? (
              <button onClick={stop} className="btn btn-danger">Stop</button>
            ) : (
              <button onClick={run} className="btn btn-primary">Run index curl</button>
            )}
            <select value={speed} onChange={(e) => setSpeed(+e.target.value)} className="field">
              {[1, 2, 4].map((s) => <option key={s} value={s} className="text-black">{s}× speed</option>)}
            </select>
            <button onClick={() => setSeed((s) => s + 1)} className="btn">New patient</button>
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

        {snap && <LivePanel snap={snap} running={!!live} />}

        <Section id="signals" title={`${sel} · week ${week} · displacement x(t) and force F(t)`}
          note="Shaded: flexion (blue), hold (amber), extension (green). Filtered, force baseline zeroed over first 0.2 s.">
          <LineChart h={110} yLabel="x (mm)" series={[{ name: "x", color: FINGER_COLOR[sel], xs: view.aI.t, ys: view.aI.xf }]}
            bands={bands} cursor={live?.finger === sel ? live.t : undefined} />
          <LineChart h={110} yLabel="F (N)" xLabel="time (s)" series={[{ name: "F", color: "#e5e7eb", xs: view.aI.t, ys: view.aI.Ff }]}
            bands={bands} cursor={live?.finger === sel ? live.t : undefined} />
          <div className="t-small muted">
            {view.aI.ok ? `${view.aI.reps.length} clean reps` : `Trial rejected: ${view.aI.reasons.join("; ")}`}
          </div>
        </Section>

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
