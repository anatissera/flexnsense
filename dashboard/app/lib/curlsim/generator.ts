import { gaussian, mulberry32 } from "./dsp";
import { Finger, FINGERS, Hand, TestKind, Trial } from "./types";

export const FS = 100;

/** Normative contralateral behaviour per finger. rom = curl path mm, fpk = peak tip force N, tf = flexion time s. */
const REF: Record<Finger, { rom: number; fpk: number; tf: number }> = {
  thumb: { rom: 80, fpk: 30, tf: 0.75 },
  index: { rom: 100, fpk: 28, tf: 0.7 },
  middle: { rom: 105, fpk: 30, tf: 0.7 },
  ring: { rom: 95, fpk: 22, tf: 0.72 },
  little: { rom: 80, fpk: 14, tf: 0.75 },
};
export const refOf = (f: Finger) => REF[f];

/** Healing course of each flexor tendon unit: capacity(t) = rinf * (1 - exp(-t/tau)). */
export interface Healing { rinf: number; tau: number }
export interface PatientProfile {
  id: string;
  seed: number;
  healing: Record<Finger, { fdp: Healing; fds: Healing }>;
}

export function makePatient(seed: number): PatientProfile {
  const rng = mulberry32(seed * 7919 + 13);
  const jit = (v: number, s: number) => v * (1 + s * gaussian(rng));
  const mk = (rinf: number, tau: number): Healing => ({
    rinf: Math.min(Math.max(jit(rinf, 0.04), 0.4), 1.05),
    tau: Math.max(jit(tau, 0.15), 1.5),
  });
  return {
    id: `P${seed}`,
    seed,
    healing: {
      thumb: { fdp: mk(0.97, 4), fds: mk(0, 1) },
      index: { fdp: mk(0.88, 9), fds: mk(0.95, 6) },
      middle: { fdp: mk(0.93, 7), fds: mk(0.97, 5) },
      ring: { fdp: mk(0.75, 14), fds: mk(0.9, 8) },
      little: { fdp: mk(0.8, 12), fds: mk(0.85, 10) },
    },
  };
}

const cap = (h: Healing, week: number) => Math.max(0.04, h.rinf * (1 - Math.exp(-week / h.tau)));
const minJerk = (u: number) => {
  const c = Math.min(Math.max(u, 0), 1);
  return 10 * c ** 3 - 15 * c ** 4 + 6 * c ** 5;
};

/** True (noise-free) structure capacity 0..1 at a given week; used only to colour the 3D model's ground truth. */
export function trueCapacity(p: PatientProfile, finger: Finger, week: number) {
  const h = p.healing[finger];
  return { fdp: cap(h.fdp, week), fds: finger === "thumb" ? null : cap(h.fds, week) };
}

export function genTrial(
  p: PatientProfile, finger: Finger, hand: Hand, session: number,
  trialIndex: number, test: TestKind, nReps = 4,
): Trial {
  const fi = FINGERS.indexOf(finger);
  const rng = mulberry32(p.seed * 100003 + fi * 9973 + session * 131 + trialIndex * 17 + (hand === "injured" ? 1 : 2) + (test === "fds" ? 555 : 0));
  const ref = REF[finger];
  const thumb = finger === "thumb";
  const wP = thumb ? 1 : 0.6;
  const wS = thumb ? 0 : 0.4;
  const capP = hand === "injured" ? cap(p.healing[finger].fdp, session) : 1 + 0.03 * gaussian(rng);
  const capS = hand === "injured" ? cap(p.healing[finger].fds, session) : 1 + 0.03 * gaussian(rng);
  const fds = test === "fds";
  const capComb = fds ? capS : wP * capP + wS * capS;
  const forceScale = fds ? wS * capS : wP * capP + wS * capS;
  const romBase = ref.rom * (fds ? 0.6 : 1) * (0.25 + 0.75 * Math.min(capComb, 1.05) ** 0.6);
  const tfBase = ref.tf / (0.35 + 0.65 * Math.min(capComb, 1.05) ** 0.7);
  const bad = 1 - Math.min(capComb, 1);
  const hyst = 0.12 + 0.35 * bad;
  const tauRelax = 2.5 * (0.5 + 0.5 * Math.min(capComb, 1));
  const relaxDrop = 0.12 + 0.2 * bad;
  const fatigue = 0.04 + 0.12 * bad;
  const hold = 1.2;
  const gap = 0.6;
  const lead = 0.6;

  const segs: { n: number; rom: number; tf: number; te: number; fpk: number }[] = [];
  let total = lead;
  for (let k = 0; k < nReps; k++) {
    const rom = romBase * (1 + (0.015 + 0.06 * bad) * gaussian(rng));
    const tf = tfBase * (1 + (0.02 + 0.08 * bad) * gaussian(rng));
    const fpk = ref.fpk * forceScale * (1 + (0.04 + 0.08 * bad) * gaussian(rng)) * (1 - (fatigue * k) / Math.max(nReps - 1, 1));
    const te = tf * 1.1;
    segs.push({ n: k, rom, tf, te, fpk });
    total += tf + hold + te + gap;
  }
  const n = Math.round(total * FS);
  const t = new Float64Array(n);
  const x = new Float64Array(n);
  const F = new Float64Array(n);
  const tremorAmp = 0.15 + 1.2 * bad;
  const tremorF = 7 + 2 * rng();
  const driftSlope = 0.04 * gaussian(rng) + (rng() < 0.04 ? 1.2 : 0);
  const offset = 0.3 + 0.2 * gaussian(rng);
  let tc = lead;
  const starts = segs.map((s) => {
    const a = tc;
    tc += s.tf + hold + s.te + gap;
    return a;
  });
  for (let i = 0; i < n; i++) {
    const ti = i / FS;
    t[i] = ti;
    let xv = 0;
    let fv = 0;
    for (let k = 0; k < segs.length; k++) {
      const s = segs[k];
      const u = ti - starts[k];
      if (u < 0 || u > s.tf + hold + s.te) continue;
      const fEnd = s.fpk * (1 - relaxDrop * (1 - Math.exp(-hold / tauRelax)));
      if (u < s.tf) {
        const sh = minJerk(u / s.tf);
        xv = s.rom * sh;
        fv = s.fpk * sh ** 1.4;
      } else if (u < s.tf + hold) {
        const th = u - s.tf;
        xv = s.rom;
        fv = s.fpk * (1 - relaxDrop * (1 - Math.exp(-th / tauRelax)));
      } else {
        const sh = 1 - minJerk((u - s.tf - hold) / s.te);
        xv = s.rom * sh;
        fv = fEnd * sh ** 1.4 * (1 - hyst * Math.sin(Math.PI * sh));
      }
    }
    x[i] = xv + tremorAmp * Math.sin(2 * Math.PI * tremorF * ti) * (xv > 0 ? 1 : 0.2) + 0.25 * gaussian(rng);
    F[i] = fv + offset + driftSlope * ti * 0.2 + 0.08 * gaussian(rng);
  }
  return { patientId: p.id, finger, hand, session, trialIndex, test, fs: FS, t, x, F };
}
