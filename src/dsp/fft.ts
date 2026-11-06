// Complex FFT of any size: fft.js (radix-4, powers of two) + Bluestein for 960/640/320 (4band_v2 n_fft).
import FFTJS from 'fft.js';

export class Fft {
  readonly n: number;
  private pow2: FFTJS | null = null;
  private m = 0;
  private big!: FFTJS;
  private chirp!: Float64Array; // w[k] = exp(-i*pi*k^2/n), interleaved
  private bHat!: number[]; // FFT of the conjugate chirp, length m
  private a: number[] = [];
  private aHat: number[] = [];
  private tmp: number[] = [];
  private io: number[] = [];
  private ioOut: number[] = [];

  constructor(n: number) {
    this.n = n;
    if ((n & (n - 1)) === 0 && n > 1) {
      this.pow2 = new FFTJS(n);
      this.io = this.pow2.createComplexArray();
      this.ioOut = this.pow2.createComplexArray();
      return;
    }
    let m = 1;
    while (m < 2 * n - 1) m <<= 1;
    this.m = m;
    this.big = new FFTJS(m);
    this.chirp = new Float64Array(2 * n);
    for (let k = 0; k < n; k++) {
      const ang = (-Math.PI * ((k * k) % (2 * n))) / n; // k^2 mod 2n keeps the angle exact
      this.chirp[2 * k] = Math.cos(ang);
      this.chirp[2 * k + 1] = Math.sin(ang);
    }
    const b = this.big.createComplexArray() as number[];
    b.fill(0);
    for (let k = 0; k < n; k++) {
      const re = this.chirp[2 * k], im = -this.chirp[2 * k + 1];
      b[2 * k] = re; b[2 * k + 1] = im;
      if (k) { b[2 * (m - k)] = re; b[2 * (m - k) + 1] = im; }
    }
    this.bHat = this.big.createComplexArray();
    this.big.transform(this.bHat, b);
    this.a = this.big.createComplexArray();
    this.aHat = this.big.createComplexArray();
    this.tmp = this.big.createComplexArray();
  }

  /** Forward DFT of interleaved complex `x` (length 2n) into `out` (length 2n). */
  forward(x: ArrayLike<number>, out: Float64Array): void {
    const n = this.n;
    if (this.pow2) {
      for (let i = 0; i < 2 * n; i++) this.io[i] = x[i];
      this.pow2.transform(this.ioOut, this.io);
      for (let i = 0; i < 2 * n; i++) out[i] = this.ioOut[i];
      return;
    }
    const { a, chirp: w, m } = this;
    a.fill(0);
    for (let k = 0; k < n; k++) {
      const xr = x[2 * k], xi = x[2 * k + 1], wr = w[2 * k], wi = w[2 * k + 1];
      a[2 * k] = xr * wr - xi * wi;
      a[2 * k + 1] = xr * wi + xi * wr;
    }
    this.big.transform(this.aHat, a);
    const A = this.aHat, B = this.bHat;
    for (let k = 0; k < m; k++) {
      const ar = A[2 * k], ai = A[2 * k + 1], br = B[2 * k], bi = B[2 * k + 1];
      A[2 * k] = ar * br - ai * bi;
      A[2 * k + 1] = ar * bi + ai * br;
    }
    this.big.inverseTransform(this.tmp, A);
    for (let k = 0; k < n; k++) {
      const cr = this.tmp[2 * k], ci = this.tmp[2 * k + 1], wr = w[2 * k], wi = w[2 * k + 1];
      out[2 * k] = cr * wr - ci * wi;
      out[2 * k + 1] = cr * wi + ci * wr;
    }
  }

  /** Inverse DFT (normalized by 1/n), via conj(DFT(conj(X))) / n. */
  inverse(X: Float64Array, out: Float64Array): void {
    const n = this.n;
    for (let i = 1; i < 2 * n; i += 2) X[i] = -X[i];
    this.forward(X, out);
    for (let i = 0; i < 2 * n; i += 2) { out[i] /= n; out[i + 1] = -out[i + 1] / n; }
  }
}

const plans = new Map<number, Fft>();
export const fftPlan = (n: number) => {
  let p = plans.get(n);
  if (!p) plans.set(n, (p = new Fft(n)));
  return p;
};
