import type { Study } from "./curlsim/study";
import { FINGERS, FLEXOR_MUSCLES, type Finger } from "./curlsim/types";
import { FACES, TURNS, VIEW_IDS } from "./viewPresets";

export interface MuscleInfo { name: string; label: string; role: string; finger?: Finger }

const F = ["", "", "index", "middle", "ring", "little"] as const;
const digitName = (d: number) => F[d];

function build(): MuscleInfo[] {
  const out: MuscleInfo[] = [];
  for (let d = 2; d <= 5; d++) {
    const f = digitName(d) as Finger;
    out.push({ name: `FDS${d}_r`, label: `Flexor digitorum superficialis (${f})`, role: `Superficial finger flexor; flexes the PIP joint (and MCP) of the ${f} finger.`, finger: f });
    out.push({ name: `FDP${d}_r`, label: `Flexor digitorum profundus (${f})`, role: `Deep finger flexor; flexes the DIP joint (and PIP/MCP) of the ${f} finger.`, finger: f });
    out.push({ name: `EDC${d}_r`, label: `Extensor digitorum communis (${f})`, role: `Extends the ${f} finger (MCP, and IP joints via the extensor hood).`, finger: f });
    out.push({ name: `RI${d}_r`, label: `Radial interosseous (${f})`, role: `Intrinsic; flexes MCP, extends IP joints of the ${f} finger.`, finger: f });
    out.push({ name: `LU_RB${d}_r`, label: `Lumbrical / radial band (${f})`, role: `Intrinsic; MCP flexion with IP extension of the ${f} finger.`, finger: f });
    out.push({ name: `UI_UB${d}_r`, label: `Ulnar interosseous / ulnar band (${f})`, role: `Intrinsic; MCP flexion with IP extension of the ${f} finger.`, finger: f });
  }
  out.push(
    { name: "FPL_r", label: "Flexor pollicis longus (thumb)", role: "Flexes the thumb IP joint (the thumb's only long flexor).", finger: "thumb" },
    { name: "EPL_r", label: "Extensor pollicis longus", role: "Extends the thumb IP joint.", finger: "thumb" },
    { name: "EPB_r", label: "Extensor pollicis brevis", role: "Extends the thumb MCP joint.", finger: "thumb" },
    { name: "APL_r", label: "Abductor pollicis longus", role: "Abducts the thumb at the carpometacarpal joint.", finger: "thumb" },
    { name: "OP_r", label: "Opponens pollicis", role: "Opposes the thumb toward the fingers.", finger: "thumb" },
    { name: "EIP_r", label: "Extensor indicis proprius", role: "Independent extension of the index finger.", finger: "index" },
    { name: "EDM_r", label: "Extensor digiti minimi", role: "Independent extension of the little finger.", finger: "little" },
    { name: "FCR_r", label: "Flexor carpi radialis", role: "Wrist flexion and radial deviation." },
    { name: "FCU_r", label: "Flexor carpi ulnaris", role: "Wrist flexion and ulnar deviation." },
    { name: "PL_r", label: "Palmaris longus", role: "Weak wrist flexor; tenses the palmar fascia." },
    { name: "ECRL_r", label: "Extensor carpi radialis longus", role: "Wrist extension and radial deviation." },
    { name: "ECRB_r", label: "Extensor carpi radialis brevis", role: "Wrist extension." },
    { name: "ECU_r", label: "Extensor carpi ulnaris", role: "Wrist extension and ulnar deviation." },
    { name: "PT_r", label: "Pronator teres", role: "Forearm pronation." },
    { name: "PQ_r", label: "Pronator quadratus", role: "Forearm pronation." },
  );
  return out;
}

export const MUSCLES: MuscleInfo[] = build();
export const MUSCLE_NAMES = MUSCLES.map((m) => m.name);
export const muscleInfo = (name: string) => MUSCLES.find((m) => m.name === name);

/** Resolve free text ("FDP ring", "FDP4", "deep flexor index") to an actuator name. */
export function resolveMuscle(q: string): string | null {
  const s = q.trim();
  const exact = MUSCLES.find((m) => m.name.toLowerCase() === s.toLowerCase() || m.name.toLowerCase() === `${s.toLowerCase()}_r`);
  if (exact) return exact.name;
  const l = s.toLowerCase();
  const finger = FINGERS.find((f) => l.includes(f)) ?? (l.includes("pinky") ? "little" : undefined);
  const kind = /fdp|profundus|deep/.test(l) ? "FDP" : /fds|superficialis|superficial/.test(l) ? "FDS" : /edc|extensor digitorum/.test(l) ? "EDC" : null;
  if (kind && finger && finger !== "thumb") return MUSCLES.find((m) => m.name.startsWith(kind) && m.finger === finger)?.name ?? null;
  if (finger === "thumb" && /flex|fpl/.test(l)) return "FPL_r";
  const digit = l.match(/(fdp|fds|edc)\s*([2-5])/);
  if (digit) return `${digit[1].toUpperCase()}${digit[2]}_r`;
  return null;
}

export const TOOLS = [
  {
    type: "function",
    function: {
      name: "highlight_muscles",
      description: "Highlight one or more muscles/tendons in the 3D scene (pulsing glow). Use whenever the user asks to see/show/find a muscle.",
      parameters: {
        type: "object",
        properties: {
          muscles: { type: "array", items: { type: "string", enum: MUSCLE_NAMES } },
        },
        required: ["muscles"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "show_muscle_info",
      description: "Show a floating info box with an arrow pointing at the muscle in 3D, listing its properties. Use when the user asks about a muscle's properties, health or recovery. Also highlights it.",
      parameters: {
        type: "object",
        properties: { muscle: { type: "string", enum: MUSCLE_NAMES } },
        required: ["muscle"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "show_dashboard_section",
      description: "Scroll the dashboard sidebar to the section that explains a metric and highlight it. Sections: live = live streaming metrics (ROM, peak force, RFD, stiffness, SPARC, hysteresis, steadiness, relaxation tau; only after Run index curl); signals = x(t)/F(t) traces with flexion/hold/extension phases; recovery_table = recovery index, R-inf, tau, t90 per finger; recovery_curve = recovery index vs week; group_scores = motion/strength/mechanics/consistency scores; force_loop = force-displacement loop (apparent stiffness, hysteresis); tendon_units = FDP/FDS estimated recovery; weights = index weights and sensitivity; data_quality = flags; export = CSV/report downloads. Use when the user asks where a metric is or to show/explain it.",
      parameters: {
        type: "object",
        properties: { section: { type: "string", enum: ["live","signals","recovery_table","recovery_curve","group_scores","force_loop","tendon_units","weights","data_quality","export"] } },
        required: ["section"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "tour_muscles",
      description: "Automatically step through every muscle/tendon popup in explore mode, one after another, highlighting each in 3D. Use when the user asks for a tour, to go through / iterate over / cycle all the muscles. Stops after the last one, or on clear_highlights / any other tool.",
      parameters: {
        type: "object",
        properties: { seconds_each: { type: "number", description: "Seconds to show each popup (default 3, 1 to 10)." } },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "set_camera_view",
      description: `Smoothly rotate the 3D camera to a named view, the same views as the view cube in the top right. The camera only rotates about the hand; it does not zoom or pan. Views: ${FACES.map((f) => `${f.id} = ${f.label}`).join("; ")}; any corner id (e.g. top_right_front) = three-quarter view from that corner; home = back to the starting side profile. "Sagittal" with no side means sagittal_right. "Frontal"/"coronal" means frontal_front. "Transverse"/"axial"/"from above"/"palm" means transverse_top; "from below"/"back of the hand" means transverse_bottom. Use when the user asks to see, show, go to, turn to or look from a view or side.`,
      parameters: {
        type: "object",
        properties: { view: { type: "string", enum: VIEW_IDS } },
        required: ["view"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "rotate_camera",
      description: "Smoothly turn the 3D camera relative to where it is now, like the arrow buttons on the view cube: up/down/left/right turn the view about the hand, roll_clockwise/roll_counterclockwise roll it about the viewing axis. Default 90 degrees. Use for relative requests such as 'rotate a bit to the left' or 'flip it upside down' (roll 180); use set_camera_view for a named view.",
      parameters: {
        type: "object",
        properties: {
          direction: { type: "string", enum: Object.keys(TURNS) },
          degrees: { type: "number", description: "How far to turn (default 90, 1 to 360)." },
        },
        required: ["direction"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "clear_highlights",
      description: "Remove all highlights and floating info boxes.",
      parameters: { type: "object", properties: {} },
    },
  },
] as const;

/** Same tools in the flat shape the Realtime API expects, plus a context getter (voice has no per-turn context). */
export const REALTIME_TOOLS = [
  ...TOOLS.map((t) => ({ type: "function" as const, ...t.function })),
  {
    type: "function" as const,
    name: "get_scene_context",
    description: "Get the current simulation/stress-test results as JSON (week, selected finger, recovery indices, tendon-unit estimates). Call before answering any question about numbers or results.",
    parameters: { type: "object", properties: {} },
  },
];

export const LANGUAGE_RULE = "Always reply in the same language the user is speaking or writing (e.g. English, French, German, Spanish, Chinese, Japanese…); switch if they switch. Muscle IDs (FDP2_r…) and units stay as written.";

const r1 = (v: number | null | undefined, k = 100) => (v == null || !Number.isFinite(v) ? null : Math.round(v * k * 10) / 10);

/** JSON-safe copy with numbers rounded to 3 places (NaN and Infinity become null), to keep the context small. */
const tidy = <T,>(x: T): unknown =>
  JSON.parse(JSON.stringify(x, (_k, v) => (typeof v === "number" ? (Number.isFinite(v) ? Math.round(v * 1000) / 1000 : null) : v instanceof Set ? undefined : v)));

/** The curl session the user has just measured (camera) or simulated, as the chat sees it. */
export interface SessionContext {
  /** "camera": the tracked finger and gauge force. "simulated": the built-in curl playback. */
  source: "camera" | "simulated";
  /** True while the session is still running: numbers and the report are provisional until it stops. */
  inProgress: boolean;
  /** Where the force came from: the gauge reader, an assumed constant, or the simulation. */
  forceSource: string;
  forceDropouts: { count: number; totalSeconds: number } | null;
  repsAnalysed: number;
  /** Fractions of the contralateral hand (0..1); the context shows them as percent. */
  recoveryIndex: number | null;
  groupScores: Record<string, number> | null;
  medianMetrics: unknown;
  trialVariability: unknown;
  perRep: unknown;
}

export function buildContext(
  study: Study, week: number, sel: Finger,
  live?: { phase: string; repsDone: number; recovery_index?: number } | null,
  session?: { measured: SessionContext | null; medicalReport: string | null },
) {
  return {
    note: "All data is SIMULATED. Recovery values are fractions of the contralateral (healthy) hand, shown here as %. 'simTruth' is the generator's hidden ground truth.",
    patient: study.patient.id,
    week,
    selectedFinger: sel,
    fingers: FINGERS.map((f) => {
      const row = study.summary.find((r) => r.finger === f && r.session === week);
      const fit = study.fits[f];
      return {
        finger: f,
        recoveryIndexPct: r1(row?.recovery_index),
        groupScoresPct: row ? Object.fromEntries(Object.entries(row.groups).map(([k, v]) => [k, r1(v)])) : null,
        medianMetrics: row?.median ?? null,
        fit: fit ? { rinfPct: r1(fit.rinf), tauWeeks: r1(fit.tau, 1), t90Weeks: Number.isFinite(fit.t90) ? r1(fit.t90, 1) : "not reached" } : null,
      };
    }),
    tendonUnits: study.structures.filter((s) => s.session === week).flatMap((s) => {
      const m = FLEXOR_MUSCLES[s.finger];
      const rows = [{ muscle: m.fdp, estPct: r1(s.fdp), simTruthPct: r1(Math.min(s.trueFdp, 1.2)) }];
      if (m.fds && s.fds != null && s.trueFds != null) rows.push({ muscle: m.fds, estPct: r1(s.fds), simTruthPct: r1(Math.min(s.trueFds, 1.2)) });
      return rows;
    }),
    weights: study.weights,
    ranking: study.sensitivity.baseRanking,
    dataQualityFlags: study.flags.slice(0, 10),
    liveAnalysis: live ?? null,
    measuredSession: session?.measured ? tidy({
      source: session.measured.source,
      inProgress: session.measured.inProgress,
      forceSource: session.measured.forceSource,
      forceDropouts: session.measured.forceDropouts,
      repsAnalysed: session.measured.repsAnalysed,
      recoveryIndexPct: r1(session.measured.recoveryIndex),
      groupScoresPct: session.measured.groupScores
        ? Object.fromEntries(Object.entries(session.measured.groupScores).map(([k, v]) => [k, r1(v)])) : null,
      medianMetrics: session.measured.medianMetrics,
      trialVariability: session.measured.trialVariability,
      perRep: session.measured.perRep,
    }) : null,
    medicalReport: session?.medicalReport ?? null,
    muscles: MUSCLES.map((m) => `${m.name}: ${m.label}`),
  };
}

export const SYSTEM_PROMPT = `You are an assistant embedded in a flexor-tendon recovery simulator (MuJoCo MyoHand 3D model + a stress-test/curl analysis).
Answer questions about the scene and the study results using ONLY the JSON context provided; if something isn't in it, say so. Be concise.
The context includes the session the user just measured: "measuredSession" (per-rep and median metrics, group scores, force source and gauge dropouts) and "medicalReport" (the assessment report in Markdown, summarising that session). Both are null until at least one curl repetition has been analysed. Use them for questions about the measured finger curls, the report, or "how did I do". If measuredSession.inProgress is true the session is still running, so say the figures are provisional. If forceSource is not the gauge, say force values are assumed rather than measured; force dropouts were recorded as 0 N.
All data is simulated; do not give clinical advice. ${LANGUAGE_RULE}
When the user asks to see, find, or learn about a specific muscle or tendon (or its properties/recovery), CALL the tools: show_muscle_info for a property question, highlight_muscles to just show muscles, show_dashboard_section to point at the dashboard section for a metric, clear_highlights to reset, tour_muscles to automatically step through every muscle popup, set_camera_view to go to a named view (sagittal, frontal, transverse, a corner, or home) and rotate_camera for a relative turn. Then briefly explain in text. Whenever your answer is about, or names, a specific muscle or tendon (even a general question like "what does the FDP do?" or "how is the index FDS healing?"), also call highlight_muscles (or show_muscle_info) for the relevant muscle(s) so it lights up in the 3D view; do not wait to be asked to show it. Skip highlighting only when the answer isn't about a particular muscle. Muscle names: FDP = deep flexor, FDS = superficial flexor, digits 2-5 = index, middle, ring, little.`;
