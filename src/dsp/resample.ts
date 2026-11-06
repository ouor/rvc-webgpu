// scipy.signal.resample_poly(x, up, down) with the default window ('kaiser', 5.0) and padtype 'constant'.
// Same filter design and output alignment as scipy/signal/_signaltools.py, so fixtures compare 1:1.

const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);

function besselI0(x: number): number {
  let sum = 1, term = 1;
  for (let k = 1; k < 200; k++) {
    term *= (x / (2 * k)) ** 2;
    sum += term;
    if (term < sum * 1e-17) break;
  }
  return sum;
}

/** scipy.signal.firwin(numtaps, cutoff, window=('kaiser', beta)) low-pass, scaled to unit DC gain. */
export function firwinKaiser(numtaps: number, cutoff: number, beta = 5.0): Float64Array {
  const h = new Float64Array(numtaps);
  const alpha = (numtaps - 1) / 2;
  const i0b = besselI0(beta);
  let s = 0;
  for (let n = 0; n < numtaps; n++) {
    const x = cutoff * (n - alpha);
    const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    const r = (2 * n) / (numtaps - 1) - 1;
    const w = besselI0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / i0b;
    h[n] = cutoff * sinc * w;
    s += h[n];
  }
  for (let n = 0; n < numtaps; n++) h[n] /= s;
  return h;
}

const filters = new Map<string, { h: Float64Array; preRemove: number }>();

/** Filter with scipy's leading zero padding; scipy's trailing zeros contribute nothing and are skipped. */
function design(up: number, down: number) {
  const key = `${up}/${down}`;
  let f = filters.get(key);
  if (!f) {
    const maxRate = Math.max(up, down);
    const halfLen = 10 * maxRate;
    const h0 = firwinKaiser(2 * halfLen + 1, 1 / maxRate);
    const pre = down - (halfLen % down);
    const h = new Float64Array(pre + h0.length);
    for (let i = 0; i < h0.length; i++) h[pre + i] = h0[i] * up;
    f = { h, preRemove: Math.floor((halfLen + pre) / down) };
    filters.set(key, f);
  }
  return f;
}

export function resamplePoly(x: ArrayLike<number>, up: number, down: number): Float32Array {
  const g = gcd(up, down);
  up /= g;
  down /= g;
  const nIn = x.length;
  if (up === 1 && down === 1) return Float32Array.from(x);
  const nOut = Math.floor((nIn * up) / down) + ((nIn * up) % down ? 1 : 0);
  const { h, preRemove } = design(up, down);
  const L = h.length;
  const y = new Float32Array(nOut);
  for (let k = 0; k < nOut; k++) {
    const t = (k + preRemove) * down; // index in the zero-stuffed signal
    const iEnd = Math.min(nIn - 1, Math.floor(t / up));
    let acc = 0;
    for (let i = Math.max(0, Math.ceil((t - L + 1) / up)); i <= iEnd; i++) acc += x[i] * h[t - i * up];
    y[k] = acc;
  }
  return y;
}

/** Resample between integer rates, e.g. resample(x, 44100, 16000). */
export const resample = (x: ArrayLike<number>, from: number, to: number) => resamplePoly(x, to, from);
