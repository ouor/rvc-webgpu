// scipy.signal.filtfilt(b, a, x) with defaults (padtype 'odd', padlen 3*max(len(a), len(b)), method 'pad').

// scipy.signal.butter(N=5, Wn=48, btype="high", fs=16000), rvc-nano pipeline.py (bh, ah).
export const HPF48_B = [0.9699606451838447, -4.849803225919223, 9.699606451838447, -9.699606451838447, 4.849803225919223, -0.9699606451838447];
export const HPF48_A = [1.0, -4.939001819168364, 9.757863526739543, -9.639544849413458, 4.761506797356209, -0.9408236532054606];

/** scipy.signal.lfilter_zi: steady-state initial conditions for a unit step (a[0] = 1). */
export function lfilterZi(b: number[], a: number[]): Float64Array {
  const n = Math.max(a.length, b.length) - 1;
  // (I - companion(a)^T) zi = b[1:] - a[1:] * b[0]
  const M = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let i = 0; i < n; i++) M[i][0] += a[i + 1];
  for (let i = 0; i + 1 < n; i++) M[i][i + 1] -= 1;
  const r = Array.from({ length: n }, (_, i) => (b[i + 1] ?? 0) - a[i + 1] * b[0]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let i = c + 1; i < n; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
    [M[c], M[p]] = [M[p], M[c]];
    [r[c], r[p]] = [r[p], r[c]];
    for (let i = c + 1; i < n; i++) {
      const f = M[i][c] / M[c][c];
      for (let j = c; j < n; j++) M[i][j] -= f * M[c][j];
      r[i] -= f * r[c];
    }
  }
  const zi = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = r[i];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * zi[j];
    zi[i] = s / M[i][i];
  }
  return zi;
}

/** In-place direct form II transposed filter; z is the initial state (consumed). */
function lfilter(b: number[], a: number[], x: Float64Array, z: Float64Array): void {
  const n = z.length;
  for (let k = 0; k < x.length; k++) {
    const xi = x[k];
    const y = b[0] * xi + z[0];
    for (let i = 0; i < n - 1; i++) z[i] = b[i + 1] * xi + z[i + 1] - a[i + 1] * y;
    z[n - 1] = b[n] * xi - a[n] * y;
    x[k] = y;
  }
}

export function filtfilt(b: number[], a: number[], x: ArrayLike<number>): Float64Array {
  const B = b.map((v) => v / a[0]), A = a.map((v) => v / a[0]);
  const len = x.length;
  const padlen = 3 * Math.max(A.length, B.length);
  if (len <= padlen) throw new Error(`filtfilt: input length ${len} must exceed padlen ${padlen}`);
  const ext = new Float64Array(len + 2 * padlen);
  for (let i = 0; i < padlen; i++) ext[i] = 2 * x[0] - x[padlen - i];
  for (let i = 0; i < len; i++) ext[padlen + i] = x[i];
  for (let i = 0; i < padlen; i++) ext[padlen + len + i] = 2 * x[len - 1] - x[len - 2 - i];
  const zi = lfilterZi(B, A);
  lfilter(B, A, ext, zi.map((v) => v * ext[0]));
  ext.reverse();
  lfilter(B, A, ext, zi.map((v) => v * ext[0]));
  ext.reverse();
  return ext.slice(padlen, padlen + len);
}
