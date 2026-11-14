import { iqr, linfit, lowpass, mean, median, sparc, std } from "./dsp";
import { Trial } from "./types";

export interface Params { fcX: number; fcF: number; maxDriftFrac: number; minReps: number }
export const DEFAULT_PARAMS: Params = { fcX: 10, fcF: 20, maxDriftFrac: 0.1, minReps: 3 };

export interface RepSeg { s: number; flexEnd: number; extStart: number; e: number }
export interface RepMetrics {
  rep: number;
  rom: number; curl_time: number; peak_velocity: number; sparc: number; norm_jerk: number;
  peak_force: number; time_to_peak: number; rfd_100: number; rfd_200: number;
  steadiness_cv: number; force_relaxation_tau: number;
  stiffness: number; work_loading: number; hysteresis: number;
  f25: number; f50: number; f75: number;
}
export const REP_KEYS: (keyof Omit<RepMetrics, "rep">)[] = [
  "rom", "curl_time", "peak_velocity", "sparc", "norm_jerk",
  "peak_force", "time_to_peak", "rfd_100", "rfd_200", "steadiness_cv", "force_relaxation_tau",
  "stiffness", "work_loading", "hysteresis", "f25", "f50", "f75",
];
export interface TrialMetrics { rom_cv: number; curl_time_cv: number; force_decay: number }

export interface TrialAnalysis {
  ok: boolean;
  reasons: string[];
  t: Float64Array; xf: Float64Array; Ff: Float64Array;
  segs: RepSeg[];
  reps: RepMetrics[];
  trial: TrialMetrics;
}

export function validate(tr: Trial) {
  for (let i = 0; i < tr.t.length; i++) {
    if (!Number.isFinite(tr.x[i]) || !Number.isFinite(tr.F[i])) throw new Error(`NaN in trial ${tr.finger}/${tr.session}`);
    if (i && tr.t[i] <= tr.t[i - 1]) throw new Error("time stamps not monotonic");
  }
  if (Math.max(...tr.F) > 500 || Math.max(...tr.x) > 400) throw new Error("implausible range");
}

const cv = (a: number[]) => (a.length > 1 ? (std(a) / mean(a)) * 100 : NaN);

/** Metrics of one repetition window g (indices into xf/Ff). Also used by the live analyzer on partial reps. */
export function computeRep(
  k: number, g: RepSeg, xf: Float64Array, Ff: Float64Array, t: Float64Array, fs: number, refRom?: number,
): RepMetrics {
  const pad = Math.round(0.3 * fs);
    const lo = Math.max(0, g.s - pad);
    const hi = Math.min(xf.length - 1, g.e + pad);
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = lo; i <= hi; i++) { mn = Math.min(mn, xf[i]); mx = Math.max(mx, xf[i]); }
    const rom = mx - mn;
    const curl_time = (g.flexEnd - g.s) / fs;
    // speed over the flexion phase
    const speed: number[] = [];
    for (let i = g.s; i <= g.flexEnd; i++) {
      const d = (xf[Math.min(i + 1, xf.length - 1)] - xf[Math.max(i - 1, 0)]) * fs / 2;
      speed.push(Math.abs(d));
    }
    const peak_velocity = Math.max(...speed);
    const sp = sparc(speed, fs);
    // normalized jerk (dimensionless)
    let j2 = 0;
    for (let i = g.s + 2; i < g.flexEnd - 1; i++) {
      const j = (xf[i + 2] - 2 * xf[i + 1] + 2 * xf[i - 1] - xf[i - 2]) * fs ** 3 / 2;
      j2 += j * j / fs;
    }
    const T = Math.max(curl_time, 1e-3);
    const norm_jerk = Math.sqrt(0.5 * j2 * T ** 5 / Math.max(rom * rom, 1e-9));

    // force
    let ip = g.s;
    for (let i = g.s; i <= g.e; i++) if (Ff[i] > Ff[ip]) ip = i;
    const peak_force = Ff[ip];
    let on = g.s;
    while (on < ip && Ff[on] < 0.1 * peak_force) on++;
    const time_to_peak = (ip - on) / fs;
    const slope = (ms: number) => {
      const k2 = Math.min(on + Math.round(ms * fs / 1000), Ff.length - 1);
      return (Ff[k2] - Ff[on]) / ((k2 - on) / fs || 1);
    };
    const hs = g.flexEnd;
    const he = g.extStart;
    let steadiness_cv = NaN;
    let force_relaxation_tau = NaN;
    if (he - hs >= 10) {
      const a = hs + Math.round(0.2 * (he - hs));
      const b = hs + Math.round(0.8 * (he - hs));
      steadiness_cv = cv(Array.from(Ff.subarray(a, b)));
      const tt: number[] = [];
      const lf: number[] = [];
      for (let i = hs; i <= he; i++) if (Ff[i] > 0.05) { tt.push(t[i] - t[hs]); lf.push(Math.log(Ff[i])); }
      const { slope: m } = linfit(tt, lf);
      force_relaxation_tau = Number.isFinite(m) ? (m < -1e-3 ? Math.min(-1 / m, 100) : 100) : NaN;
    }
    // force-displacement
    const fx: number[] = [];
    const ff: number[] = [];
    for (let i = g.s; i <= g.flexEnd; i++) {
      const u = (xf[i] - mn) / rom;
      if (u >= 0.2 && u <= 0.8) { fx.push(xf[i]); ff.push(Ff[i]); }
    }
    const stiffness = linfit(fx, ff).slope;
    let wl = 0;
    for (let i = g.s; i < g.flexEnd; i++) wl += 0.5 * (Ff[i] + Ff[i + 1]) * (xf[i + 1] - xf[i]);
    let wu = 0;
    for (let i = g.extStart; i < g.e; i++) wu += 0.5 * (Ff[i] + Ff[i + 1]) * (xf[i] - xf[i + 1]);
    const hysteresis = wl > 0 ? ((wl - wu) / wl) * 100 : NaN;
    const rr = refRom ?? rom;
    const fAt = (frac: number) => {
      const d = frac * rr + mn;
      for (let i = g.s; i < g.flexEnd; i++)
        if (xf[i + 1] >= d && xf[i] < d) {
          const w = (d - xf[i]) / (xf[i + 1] - xf[i]);
          return Ff[i] + w * (Ff[i + 1] - Ff[i]);
        }
      return xf[g.flexEnd] >= d ? Ff[g.flexEnd] : 0; // posture not reached => no force there
    };
    return {
      rep: k + 1, rom, curl_time, peak_velocity, sparc: sp, norm_jerk, peak_force, time_to_peak,
      rfd_100: slope(100), rfd_200: slope(200), steadiness_cv, force_relaxation_tau,
      stiffness, work_loading: wl, hysteresis, f25: fAt(0.25), f50: fAt(0.5), f75: fAt(0.75),
    };
}

/** refRom: reference ROM (mm) for force_at_displacement; defaults to this trial's own median. */
export function analyzeTrial(tr: Trial, refRom?: number, p: Params = DEFAULT_PARAMS): TrialAnalysis {
  validate(tr);
  const { fs, t } = tr;
  const n0 = Math.round(0.2 * fs);
  const xf = lowpass(tr.x, fs, p.fcX);
  const Ff = lowpass(tr.F, fs, p.fcF);
  const x0 = mean(xf.subarray(0, n0));
  const F0 = mean(Ff.subarray(0, n0));
  for (let i = 0; i < xf.length; i++) { xf[i] -= x0; Ff[i] -= F0; }

  const reasons: string[] = [];
  const xmin = Math.min(...xf);
  const xmax = Math.max(...xf);
  const range = xmax - xmin;
  const segs: RepSeg[] = [];
  if (range < 3) reasons.push("no curl detected (range < 3 mm)");
  else {
    const thr = xmin + 0.05 * range;
    let s = -1;
    for (let i = 1; i < xf.length; i++) {
      if (s < 0 && xf[i] > thr && xf[i - 1] <= thr) s = i;
      else if (s >= 0 && xf[i] < thr) {
        let pk = s;
        for (let j = s; j <= i; j++) if (xf[j] > xf[pk]) pk = j;
        const lvl = xmin + 0.95 * (xf[pk] - xmin);
        let fe = s;
        while (fe < i && xf[fe] < lvl) fe++;
        let es = i;
        while (es > fe && xf[es] < lvl) es--;
        segs.push({ s, flexEnd: fe, extStart: es, e: i });
        s = -1;
      }
    }
  }
  if (segs.length < p.minReps) reasons.push(`only ${segs.length} clean reps (< ${p.minReps})`);
  const tail = mean(Ff.subarray(Ff.length - n0));
  const peakGuess = Math.max(...Ff);
  if (peakGuess > 0 && Math.abs(tail) > p.maxDriftFrac * peakGuess)
    reasons.push(`baseline drift ${tail.toFixed(2)} N (> ${(p.maxDriftFrac * 100).toFixed(0)}% of peak)`);

  const reps: RepMetrics[] = segs.map((g, k) => computeRep(k, g, xf, Ff, t, fs, refRom));
  const pf = reps.map((r) => r.peak_force);
  const trial: TrialMetrics = {
    rom_cv: cv(reps.map((r) => r.rom)),
    curl_time_cv: cv(reps.map((r) => r.curl_time)),
    force_decay: pf.length > 1 ? ((pf[0] - pf[pf.length - 1]) / pf[0]) * 100 : NaN,
  };
  return { ok: reasons.length === 0, reasons, t, xf, Ff, segs, reps, trial };
}

export const medIqr = (v: number[]) => ({ median: median(v), iqr: iqr(v) });
