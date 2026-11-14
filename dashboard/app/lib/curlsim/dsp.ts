// Small numerical helpers: RNG, zero-phase Butterworth, FFT, SPARC, stats.

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng: () => number) {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

function biquad(x: Float64Array, b: number[], a: number[]) {
  const y = new Float64Array(x.length);
  let z1 = 0;
  let z2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = b[0] * x[i] + z1;
    z1 = b[1] * x[i] - a[1] * v + z2;
    z2 = b[2] * x[i] - a[2] * v;
    y[i] = v;
  }
  return y;
}

/** Zero-phase 2nd-order Butterworth low-pass (forward-backward => 4th order). */
export function lowpass(x: Float64Array, fs: number, fc: number): Float64Array {
  const fcEff = Math.min(fc, 0.45 * fs);
  const K = Math.tan((Math.PI * fcEff) / fs);
  const norm = 1 / (1 + Math.SQRT2 * K + K * K);
  const b0 = K * K * norm;
  const b = [b0, 2 * b0, b0];
  const a = [1, 2 * (K * K - 1) * norm, (1 - Math.SQRT2 * K + K * K) * norm];
  const n = x.length;
  const pad = Math.min(Math.ceil((3 * fs) / fcEff), n - 1);
  const ext = new Float64Array(n + 2 * pad);
  for (let i = 0; i < pad; i++) {
    ext[i] = 2 * x[0] - x[pad - i];
    ext[n + pad + i] = 2 * x[n - 1] - x[n - 2 - i];
  }
  ext.set(x, pad);
  const f = biquad(ext, b, a);
  f.reverse();
  const r = biquad(f, b, a);
  r.reverse();
  return r.slice(pad, pad + n);
}

export function fftMag(x: ArrayLike<number>, nfft: number): Float64Array {
  const re = new Float64Array(nfft);
  const im = new Float64Array(nfft);
  for (let i = 0; i < x.length; i++) re[i] = x[i];
  for (let i = 1, j = 0; i < nfft; i++) {
    let bit = nfft >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
    }
  }
  for (let len = 2; len <= nfft; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < nfft; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const u = i + k;
        const v = i + k + len / 2;
        const tr = re[v] * cr - im[v] * ci;
        const ti = re[v] * ci + im[v] * cr;
        re[v] = re[u] - tr;
        im[v] = im[u] - ti;
        re[u] += tr;
        im[u] += ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
  const m = new Float64Array(nfft / 2 + 1);
  for (let i = 0; i < m.length; i++) m[i] = Math.hypot(re[i], im[i]);
  return m;
}

/** Spectral arc length of a speed profile (Balasubramanian et al. 2015). More negative = less smooth. */
export function sparc(speed: ArrayLike<number>, fs: number, fc = 10, padLevel = 4, ampTh = 0.05) {
  if (speed.length < 8) return NaN;
  const nfft = 2 ** (Math.ceil(Math.log2(speed.length)) + padLevel);
  const mag = fftMag(speed, nfft);
  const df = fs / nfft;
  const peak = Math.max(...mag);
  if (peak <= 0) return NaN;
  const nMax = Math.min(Math.floor(fc / df), mag.length - 1);
  let last = 0;
  for (let i = 0; i <= nMax; i++) if (mag[i] / peak >= ampTh) last = i;
  if (last < 1) return NaN;
  let arc = 0;
  for (let i = 1; i <= last; i++) {
    const d = (mag[i] - mag[i - 1]) / peak;
    arc += Math.hypot(1 / last, d);
  }
  return -arc;
}

export const mean = (a: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return a.length ? s / a.length : NaN;
};
export const std = (a: ArrayLike<number>) => {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2;
  return a.length > 1 ? Math.sqrt(s / (a.length - 1)) : NaN;
};
export function quantile(a: number[], q: number) {
  const v = a.filter(Number.isFinite).sort((p, r) => p - r);
  if (!v.length) return NaN;
  const pos = (v.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}
export const median = (a: number[]) => quantile(a, 0.5);
export const iqr = (a: number[]) => quantile(a, 0.75) - quantile(a, 0.25);

export function linfit(xs: number[], ys: number[]) {
  const n = xs.length;
  if (n < 3) return { slope: NaN, intercept: NaN };
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) ** 2;
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const slope = sxx > 0 ? sxy / sxx : NaN;
  return { slope, intercept: my - slope * mx };
}

/** Saturating recovery fit R(t) = Rinf (1 - exp(-t/tau)); grid over tau, closed-form Rinf. */
export function fitSaturating(t: number[], r: number[]) {
  const pts = t.map((ti, i) => [ti, r[i]] as const).filter(([, v]) => Number.isFinite(v));
  if (pts.length < 3) return null;
  let best: { rinf: number; tau: number; rmse: number } | null = null;
  for (let tau = 0.5; tau <= 80; tau *= 1.02) {
    let se = 0;
    let ee = 0;
    for (const [ti, v] of pts) {
      const e = 1 - Math.exp(-ti / tau);
      se += e * v;
      ee += e * e;
    }
    const rinf = Math.min(Math.max(se / ee, 0), 1.5);
    let sse = 0;
    for (const [ti, v] of pts) sse += (v - rinf * (1 - Math.exp(-ti / tau))) ** 2;
    const rmse = Math.sqrt(sse / pts.length);
    if (!best || rmse < best.rmse) best = { rinf, tau, rmse };
  }
  if (!best) return null;
  // Time to reach 90 % of the reference; NaN when the plateau is below that.
  const t90 = best.rinf > 0.9 ? -best.tau * Math.log(1 - 0.9 / best.rinf) : NaN;
  return { ...best, t90 };
}
export const curveAt = (f: { rinf: number; tau: number }, t: number) =>
  f.rinf * (1 - Math.exp(-t / f.tau));
