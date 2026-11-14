/** Single-pass (causal) 2nd-order Butterworth low-pass for streaming use; adds a small group delay. */
export class CausalLowpass {
  private b: number[];
  private a: number[];
  private z1 = 0;
  private z2 = 0;
  private primed = false;
  private fs: number;
  constructor(fs: number, fc: number) {
    this.fs = fs;
    const K = Math.tan((Math.PI * Math.min(fc, 0.45 * fs)) / fs);
    const norm = 1 / (1 + Math.SQRT2 * K + K * K);
    const b0 = K * K * norm;
    this.b = [b0, 2 * b0, b0];
    this.a = [1, 2 * (K * K - 1) * norm, (1 - Math.SQRT2 * K + K * K) * norm];
  }
  step(x: number): number {
    if (!this.primed) {
      this.primed = true;
      for (let i = 0; i < Math.ceil(this.fs); i++) this.step(x); // settle on the first sample
    }
    const v = this.b[0] * x + this.z1;
    this.z1 = this.b[1] * x - this.a[1] * v + this.z2;
    this.z2 = this.b[2] * x - this.a[2] * v;
    return v;
  }
}
