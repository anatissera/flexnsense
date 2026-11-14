import { curveAt, fitSaturating, iqr, mean, median } from "./dsp";
import { genTrial, makePatient, PatientProfile, trueCapacity } from "./generator";
import { analyzeTrial, REP_KEYS, RepMetrics, TrialAnalysis } from "./metrics";
import { Finger, FINGERS, Hand, SESSIONS, TestKind, Trial } from "./types";

export type GroupName = "motion" | "strength" | "mechanics" | "consistency";
export const GROUPS: GroupName[] = ["motion", "strength", "mechanics", "consistency"];
export type Weights = Record<GroupName, number>;
export const DEFAULT_WEIGHTS: Weights = { motion: 0.3, strength: 0.35, mechanics: 0.2, consistency: 0.15 };

interface Spec { key: string; lower: boolean; abs?: boolean; eps: number }
const GROUP_SPECS: Record<GroupName, Spec[][]> = {
  // inner arrays are averaged first (force_at_displacement = mean of its three points)
  motion: [
    [{ key: "rom", lower: false, eps: 1e-9 }],
    [{ key: "curl_time", lower: true, eps: 0.05 }],
    [{ key: "peak_velocity", lower: false, eps: 1e-9 }],
    [{ key: "sparc", lower: true, abs: true, eps: 0.1 }], // |SPARC|: more negative = less smooth
  ],
  strength: [
    [{ key: "peak_force", lower: false, eps: 1e-9 }],
    [{ key: "rfd_200", lower: false, eps: 1e-9 }],
    [{ key: "f25", lower: false, eps: 1e-9 }, { key: "f50", lower: false, eps: 1e-9 }, { key: "f75", lower: false, eps: 1e-9 }],
  ],
  mechanics: [
    [{ key: "stiffness", lower: false, eps: 1e-9 }],
    [{ key: "hysteresis", lower: true, eps: 1 }],
    [{ key: "force_relaxation_tau", lower: false, eps: 1e-9 }],
  ],
  consistency: [
    [{ key: "repeatability_cv", lower: true, eps: 1 }],
    [{ key: "steadiness_cv", lower: true, eps: 1 }],
    [{ key: "force_decay", lower: true, eps: 1 }],
  ],
};

export type Med = Record<string, number>;
export interface SummaryRow {
  patient: string; finger: Finger; session: number; hand: Hand;
  normalized: boolean; n_trials_ok: number; n_reps: number;
  median: Med; iqr: Med;
  norm: Med;
  groups: Record<GroupName, number>;
  recovery_index: number;
}
export interface RepRow extends RepMetrics { patient: string; finger: Finger; hand: Hand; session: number; trial: number }
export interface Fit { rinf: number; tau: number; rmse: number; t90: number }
export interface StructureRow { finger: Finger; session: number; fdp: number; fds: number | null; trueFdp: number; trueFds: number | null }
export interface Sensitivity {
  session: number;
  perFinger: Record<Finger, { base: number; min: number; max: number }>;
  baseRanking: Finger[];
  scenarios: { label: string; ranking: Finger[]; changed: boolean }[];
  nChanged: number;
}
export interface Study {
  patient: PatientProfile;
  weights: Weights;
  repRows: RepRow[];
  summary: SummaryRow[];
  structures: StructureRow[];
  flags: string[];
  fits: Record<Finger, Fit | null>;
  structureFits: Record<string, Fit | null>; // key `${finger}:fdp|fds`
  sensitivity: Sensitivity;
}

const cap = (v: number) => Math.min(Math.max(v, 0), 1.2);

export function normalizeMetric(m: number, r: number, s: Spec) {
  if (!Number.isFinite(m) || !Number.isFinite(r)) return NaN;
  const a = s.abs ? Math.abs(m) : m;
  const b = s.abs ? Math.abs(r) : r;
  if (b <= 0 && !s.lower) return NaN;
  const v = s.lower ? Math.max(b, s.eps) / Math.max(a, s.eps) : a / b;
  return cap(v);
}

export function scoreGroups(med: Med, ref: Med) {
  const norm: Med = {};
  const groups = {} as Record<GroupName, number>;
  for (const g of GROUPS) {
    const parts: number[] = [];
    for (const inner of GROUP_SPECS[g]) {
      const vals = inner.map((s) => {
        const v = normalizeMetric(med[s.key], ref[s.key], s);
        norm[s.key] = v;
        return v;
      }).filter(Number.isFinite);
      if (vals.length) parts.push(mean(vals));
    }
    groups[g] = parts.length ? mean(parts) : NaN;
  }
  return { norm, groups };
}

export function recoveryIndex(groups: Record<GroupName, number>, w: Weights) {
  let s = 0;
  let ws = 0;
  for (const g of GROUPS) if (Number.isFinite(groups[g])) { s += w[g] * groups[g]; ws += w[g]; }
  return ws > 0 ? s / ws : NaN;
}

interface Pool { reps: (RepMetrics & { trial: number })[]; med: Med; iqr: Med; nOk: number; flags: string[] }
function pool(trials: Trial[], refRom: number | undefined, label: string): Pool {
  const anas: TrialAnalysis[] = trials.map((t) => analyzeTrial(t, refRom));
  const flags: string[] = [];
  const ok = anas.filter((a, i) => {
    if (!a.ok) flags.push(`${label} trial ${i + 1} rejected: ${a.reasons.join("; ")}`);
    return a.ok;
  });
  const reps = ok.flatMap((a, ti) => a.reps.map((r) => ({ ...r, trial: ti + 1 })));
  const med: Med = {};
  const iq: Med = {};
  for (const k of REP_KEYS) {
    const v = reps.map((r) => r[k]);
    med[k] = median(v);
    iq[k] = iqr(v);
  }
  const romCv = ok.map((a) => a.trial.rom_cv);
  const ctCv = ok.map((a) => a.trial.curl_time_cv);
  med.rom_cv = median(romCv);
  med.curl_time_cv = median(ctCv);
  med.repeatability_cv = mean([med.rom_cv, med.curl_time_cv]);
  med.force_decay = median(ok.map((a) => a.trial.force_decay));
  iq.repeatability_cv = NaN;
  iq.force_decay = iqr(ok.map((a) => a.trial.force_decay));
  return { reps, med, iqr: iq, nOk: ok.length, flags };
}

const peakOf = (trials: Trial[]) => {
  const v = trials.flatMap((t) => analyzeTrial(t).reps.map((r) => r.peak_force));
  return median(v);
};

export function runStudy(seed: number, weights: Weights = DEFAULT_WEIGHTS, sessions = SESSIONS): Study {
  const patient = makePatient(seed);
  const repRows: RepRow[] = [];
  const summary: SummaryRow[] = [];
  const structures: StructureRow[] = [];
  const flags: string[] = [];
  const NT = 3;
  const gen = (f: Finger, h: Hand, s: number, test: TestKind, n: number) =>
    Array.from({ length: n }, (_, i) => genTrial(patient, f, h, s, i + 1, test));

  for (const finger of FINGERS) for (const session of sessions) {
    const tag = `${finger} wk${session}`;
    try {
      const contraT = gen(finger, "contralateral", session, "full", NT);
      const injT = gen(finger, "injured", session, "full", NT);
      // pass 1 gives the reference ROM used by force_at_displacement
      const ref0 = pool(contraT, undefined, `${tag} contralateral`);
      if (ref0.nOk === 0) { flags.push(`${tag}: no usable contralateral trials; row skipped`); continue; }
      const refRom = ref0.med.rom;
      const ref = pool(contraT, refRom, `${tag} contralateral`);
      const inj = pool(injT, refRom, `${tag} injured`);
      flags.push(...ref.flags, ...inj.flags);
      if (inj.nOk === 0) { flags.push(`${tag}: no usable injured trials; row skipped`); continue; }
      for (const [hand, p] of [["contralateral", ref], ["injured", inj]] as const)
        for (const r of p.reps) repRows.push({ ...r, patient: patient.id, finger, hand, session });
      const { norm, groups } = scoreGroups(inj.med, ref.med);
      summary.push({
        patient: patient.id, finger, session, hand: "injured", normalized: true,
        n_trials_ok: inj.nOk, n_reps: inj.reps.length,
        median: inj.med, iqr: inj.iqr, norm, groups, recovery_index: recoveryIndex(groups, weights),
      });
      // per-structure attribution (FDS-only DIP-blocked test vs full curl)
      const tc = trueCapacity(patient, finger, session);
      if (finger === "thumb") {
        structures.push({ finger, session, fdp: cap(inj.med.peak_force / ref.med.peak_force), fds: null, trueFdp: tc.fdp, trueFds: null });
      } else {
        const fdsC = peakOf(gen(finger, "contralateral", session, "fds", 1));
        const fdsI = peakOf(gen(finger, "injured", session, "fds", 1));
        const fdpC = Math.max(ref.med.peak_force - fdsC, 1e-3);
        const fdpI = Math.max(inj.med.peak_force - fdsI, 0);
        structures.push({ finger, session, fdp: cap(fdpI / fdpC), fds: cap(fdsI / fdsC), trueFdp: tc.fdp, trueFds: tc.fds });
      }
    } catch (e) {
      flags.push(`${tag}: skipped (${(e as Error).message})`);
    }
  }

  const fits = {} as Study["fits"];
  const structureFits: Study["structureFits"] = {};
  for (const f of FINGERS) {
    const rows = summary.filter((r) => r.finger === f);
    fits[f] = fitSaturating(rows.map((r) => r.session), rows.map((r) => r.recovery_index));
    const st = structures.filter((r) => r.finger === f);
    structureFits[`${f}:fdp`] = fitSaturating(st.map((r) => r.session), st.map((r) => r.fdp));
    if (f !== "thumb")
      structureFits[`${f}:fds`] = fitSaturating(st.map((r) => r.session), st.map((r) => r.fds ?? NaN));
  }
  return { patient, weights, repRows, summary, structures, flags, fits, structureFits, sensitivity: sensitivity(summary, weights) };
}

function ranking(vals: Record<string, number>) {
  return (Object.keys(vals) as Finger[]).sort((a, b) => vals[b] - vals[a]);
}

export function sensitivity(summary: SummaryRow[], w: Weights): Sensitivity {
  const session = Math.max(...summary.map((r) => r.session));
  const rows = summary.filter((r) => r.session === session);
  const idx = (ww: Weights) => Object.fromEntries(rows.map((r) => [r.finger, recoveryIndex(r.groups, ww)])) as Record<Finger, number>;
  const base = idx(w);
  const baseRanking = ranking(base);
  const perFinger = Object.fromEntries(rows.map((r) => [r.finger, { base: base[r.finger], min: base[r.finger], max: base[r.finger] }])) as Sensitivity["perFinger"];
  const scenarios: Sensitivity["scenarios"] = [];
  for (const g of GROUPS) for (const m of [0.5, 1.5]) {
    const ww = { ...w, [g]: w[g] * m } as Weights;
    const v = idx(ww);
    for (const f of Object.keys(v) as Finger[]) {
      perFinger[f].min = Math.min(perFinger[f].min, v[f]);
      perFinger[f].max = Math.max(perFinger[f].max, v[f]);
    }
    const rk = ranking(v);
    scenarios.push({ label: `${g} x${m}`, ranking: rk, changed: rk.join() !== baseRanking.join() });
  }
  return { session, perFinger, baseRanking, scenarios, nChanged: scenarios.filter((s) => s.changed).length };
}

export function reweight(study: Study, w: Weights): Study {
  const summary = study.summary.map((r) => ({ ...r, recovery_index: recoveryIndex(r.groups, w) }));
  const fits = {} as Study["fits"];
  for (const f of FINGERS) {
    const rows = summary.filter((r) => r.finger === f);
    fits[f] = fitSaturating(rows.map((r) => r.session), rows.map((r) => r.recovery_index));
  }
  return { ...study, weights: w, summary, fits, sensitivity: sensitivity(summary, w) };
}

// ---------- exports ----------
const fmt = (v: unknown) => (typeof v === "number" ? (Number.isFinite(v) ? +v.toFixed(5) : "") : v ?? "");
export function toCsv(rows: Record<string, unknown>[]) {
  if (!rows.length) return "";
  const cols = Object.keys(rows[0]);
  return [cols.join(","), ...rows.map((r) => cols.map((c) => fmt(r[c])).join(","))].join("\n");
}
export const metricsCsv = (s: Study) =>
  toCsv(s.repRows.map((r) => ({ ...r })));
export const summaryCsv = (s: Study) =>
  toCsv(s.summary.map((r) => {
    const o: Record<string, unknown> = { patient: r.patient, finger: r.finger, session: r.session, normalized: r.normalized, n_trials_ok: r.n_trials_ok, n_reps: r.n_reps };
    for (const k of Object.keys(r.median)) { o[`${k}_median`] = r.median[k]; if (k in r.iqr) o[`${k}_iqr`] = r.iqr[k]; }
    for (const g of GROUPS) o[`${g}_score`] = r.groups[g] * 100;
    o.recovery_index = r.recovery_index * 100;
    return o;
  }));

export function reportMd(s: Study) {
  const last = Math.max(...s.summary.map((r) => r.session));
  const now = s.summary.filter((r) => r.session === last).sort((a, b) => b.recovery_index - a.recovery_index);
  const L: string[] = [`# Recovery report - ${s.patient.id} (SIMULATED DATA)`, "", `Latest session: week ${last}. Index = weighted mean of group scores (heuristic, % of contralateral reference).`, "",
    "| Finger | Recovery index | R_inf | tau (wk) | Time to 90 % |", "|---|---|---|---|---|"];
  for (const r of now) {
    const f = s.fits[r.finger];
    L.push(`| ${r.finger} | ${(r.recovery_index * 100).toFixed(0)}% | ${f ? (f.rinf * 100).toFixed(0) + "%" : "-"} | ${f ? f.tau.toFixed(1) : "-"} | ${f ? (Number.isFinite(f.t90) ? f.t90.toFixed(1) + " wk" : "not reached") : "-"} |`);
  }
  if (now.length) L.push("", `Best finger: **${now[0].finger}**; worst finger: **${now[now.length - 1].finger}**.`);
  L.push("", "## Model-attributed flexor tendon-unit recovery (peak-force share, latest session)", "", "| Finger | FDP/FPL | FDS |", "|---|---|---|");
  for (const r of s.structures.filter((x) => x.session === last))
    L.push(`| ${r.finger} | ${(r.fdp * 100).toFixed(0)}% | ${r.fds == null ? "n/a" : (r.fds * 100).toFixed(0) + "%"} |`);
  L.push("", `## Sensitivity (weights +/-50 %, week ${s.sensitivity.session})`, "");
  for (const f of Object.keys(s.sensitivity.perFinger) as Finger[]) {
    const p = s.sensitivity.perFinger[f];
    L.push(`- ${f}: ${(p.min * 100).toFixed(0)}-${(p.max * 100).toFixed(0)}% (base ${(p.base * 100).toFixed(0)}%)`);
  }
  L.push(`- Finger ranking changed in ${s.sensitivity.nChanged} of ${s.sensitivity.scenarios.length} scenarios.`);
  L.push("", "## Data-quality flags", "", ...(s.flags.length ? s.flags.map((f) => `- ${f}`) : ["- none"]));
  L.push("", "> Apparent/effective mechanics only; flexor-driven curls only (no extensors, intrinsics, ligaments or pulleys). Research/demo use, not clinical decision-making.");
  return L.join("\n");
}

export { curveAt };
