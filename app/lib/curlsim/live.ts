import { CausalLowpass } from "./causal";
import { mean, median, std } from "./dsp";
import { computeRep, DEFAULT_PARAMS, REP_KEYS, RepMetrics, RepSeg, TrialAnalysis } from "./metrics";
import { Med, recoveryIndex, scoreGroups, Weights, type GroupName } from "./study";

export type LivePhase = "baseline" | "rest" | "flexion" | "hold" | "extension";

export interface TrialLevel { rom_cv: number; curl_time_cv: number; force_decay: number }

export interface LiveSnapshot {
  t: number;
  phase: LivePhase;
  repsDone: number;
  /** Metrics of the repetition in progress; a key is absent until it is computable. */
  current: Partial<RepMetrics>;
  /** Keys of `current` whose value will no longer change in this rep. */
  final: Set<string>;
  reps: RepMetrics[];
  trial: TrialLevel;
  med: Med;
  groups: Record<GroupName, number> | null;
  recovery_index: number | null;
  /** Streaming delay: filter group delay (s). */
  latency: number;
}

const cv = (a: number[]) => (a.length > 1 ? (std(a) / mean(a)) * 100 : NaN);

export function medOf(reps: RepMetrics[], trial: TrialLevel): Med {
  const m: Med = {};
  for (const k of REP_KEYS) m[k] = median(reps.map((r) => r[k]));
  m.rom_cv = trial.rom_cv;
  m.curl_time_cv = trial.curl_time_cv;
  m.repeatability_cv = mean([trial.rom_cv, trial.curl_time_cv].filter(Number.isFinite));
  m.force_decay = trial.force_decay;
  return m;
}
export const medOfAnalysis = (a: TrialAnalysis) => medOf(a.reps, a.trial);

/**
 * Sample-by-sample analyzer: causal filters, running baseline zeroing, a phase state machine
 * (rest -> flexion -> hold -> extension) and incremental per-rep metrics. Metrics that need the
 * whole rep (hysteresis, repeatability) only appear once the rep ends.
 */
export class LiveAnalyzer {
  private fx: CausalLowpass;
  private fF: CausalLowpass;
  private t: number[] = [];
  private x: number[] = [];
  private F: number[] = [];
  private n0: number;
  private offX = 0;
  private offF = 0;
  private phase: LivePhase = "baseline";
  private s = 0;
  private flexEnd = 0;
  private extStart = 0;
  private runMax = 0;
  private peakSpeed = 0;
  private reps: RepMetrics[] = [];
  private lastCompute = -1;
  private cache: { current: Partial<RepMetrics>; final: Set<string> } = { current: {}, final: new Set() };
  private fs: number;
  private refRom: number;
  private ref: Med | null;
  private weights: Weights;

  constructor(fs: number, refRom: number, ref: Med | null, weights: Weights) {
    this.fs = fs;
    this.refRom = refRom;
    this.ref = ref;
    this.weights = weights;
    this.fx = new CausalLowpass(fs, DEFAULT_PARAMS.fcX);
    this.fF = new CausalLowpass(fs, DEFAULT_PARAMS.fcX); // same cutoff as x so F and x stay phase-aligned (hysteresis depends on it)
    this.n0 = Math.round(0.2 * fs);
  }

  push(tSample: number, xRaw: number, FRaw: number) {
    const i = this.t.length;
    this.t.push(tSample);
    this.x.push(this.fx.step(xRaw));
    this.F.push(this.fF.step(FRaw));
    if (i + 1 === this.n0) {
      this.offX = mean(this.x);
      this.offF = mean(this.F);
      for (let k = 0; k < this.x.length; k++) { this.x[k] -= this.offX; this.F[k] -= this.offF; }
      this.phase = "rest";
    } else if (i >= this.n0) {
      this.x[i] -= this.offX;
      this.F[i] -= this.offF;
    }
    if (this.phase === "baseline") return;
    this.updatePhase(i);
  }

  private updatePhase(i: number) {
    const { fs, refRom } = this;
    const x = this.x[i];
    const thr = 0.05 * refRom;
    const v = i > 0 ? (x - this.x[i - 1]) * fs : 0;
    const margin = Math.max(0.05 * this.runMax, 1.5);
    switch (this.phase) {
      case "rest":
        if (x > thr && v > 0) {
          this.phase = "flexion"; this.s = i; this.runMax = x; this.peakSpeed = 0; this.lastCompute = -1;
          this.cache = { current: {}, final: new Set() };
        }
        break;
      case "flexion":
        this.runMax = Math.max(this.runMax, x);
        this.peakSpeed = Math.max(this.peakSpeed, Math.abs(v));
        if (x >= this.runMax - margin && this.peakSpeed > 0 && Math.abs(v) < 0.15 * this.peakSpeed && x > 0.3 * this.runMax) {
          this.phase = "hold"; this.flexEnd = i;
        }
        break;
      case "hold":
        this.runMax = Math.max(this.runMax, x);
        if (x < this.runMax - 2 * margin && v < 0) {
          // Detection lags the true release; walk back to the last sample still at ~95 % of peak.
          const lvl = 0.95 * this.runMax;
          let k = i;
          while (k > this.flexEnd && this.x[k] < lvl) k--;
          this.phase = "extension"; this.extStart = k;
        }
        break;
      case "extension":
        if (x < thr) { this.finishRep(i); this.phase = "rest"; }
        break;
    }
  }

  private arrays() {
    return [Float64Array.from(this.t), Float64Array.from(this.x), Float64Array.from(this.F)] as const;
  }

  private partial(i: number): { seg: RepSeg; final: string[] } {
    const s = this.s;
    if (this.phase === "flexion") return { seg: { s, flexEnd: i, extStart: i, e: i }, final: [] };
    if (this.phase === "hold")
      return { seg: { s, flexEnd: this.flexEnd, extStart: i, e: i }, final: ["curl_time", "peak_velocity", "sparc", "norm_jerk"] };
    return {
      seg: { s, flexEnd: this.flexEnd, extStart: this.extStart, e: i },
      final: ["curl_time", "peak_velocity", "sparc", "norm_jerk", "steadiness_cv", "force_relaxation_tau"],
    };
  }

  private finishRep(i: number) {
    const [t, x, F] = this.arrays();
    const seg = { s: this.s, flexEnd: this.flexEnd, extStart: this.extStart, e: i };
    this.reps.push(computeRep(this.reps.length, seg, x, F, t, this.fs, this.refRom));
    this.cache = { current: {}, final: new Set() };
  }

  /** Recompute provisional metrics of the rep in progress (call at UI rate, not every sample). */
  snapshot(): LiveSnapshot {
    const i = this.t.length - 1;
    const active = this.phase === "flexion" || this.phase === "hold" || this.phase === "extension";
    if (active && i !== this.lastCompute) {
      this.lastCompute = i;
      const [t, x, F] = this.arrays();
      const { seg, final } = this.partial(i);
      const cur: Partial<RepMetrics> = { ...computeRep(this.reps.length, seg, x, F, t, this.fs, this.refRom) };
      // Only keep metrics that are meaningful for the phase reached so far.
      if (this.phase !== "extension") delete cur.hysteresis;
      if (this.phase === "flexion") { delete cur.steadiness_cv; delete cur.force_relaxation_tau; }
      this.cache = { current: cur, final: new Set(final) };
    } else if (!active) {
      this.cache = { current: {}, final: new Set() };
    }
    const pf = this.reps.map((r) => r.peak_force);
    const trial: TrialLevel = {
      rom_cv: cv(this.reps.map((r) => r.rom)),
      curl_time_cv: cv(this.reps.map((r) => r.curl_time)),
      force_decay: pf.length > 1 ? ((pf[0] - pf[pf.length - 1]) / pf[0]) * 100 : NaN,
    };
    let groups: Record<GroupName, number> | null = null;
    let ri: number | null = null;
    const med = this.reps.length ? medOf(this.reps, trial) : {};
    if (this.ref && this.reps.length) {
      groups = scoreGroups(med, this.ref).groups;
      ri = recoveryIndex(groups, this.weights);
    }
    return {
      t: this.t.length ? this.t[this.t.length - 1] : 0,
      phase: this.phase, repsDone: this.reps.length,
      current: this.cache.current, final: this.cache.final,
      reps: [...this.reps], trial, med, groups, recovery_index: ri,
      latency: 1 / DEFAULT_PARAMS.fcX + 2 / this.fs,
    };
  }
}
