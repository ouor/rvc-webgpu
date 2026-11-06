// numpy.pad index mapping, shared by STFT, RMS and the RVC pipeline padding.

export type PadMode = 'reflect' | 'constant';

/** numpy.pad(mode='reflect') source index for padded position i (any pad length, also longer than n). */
export function reflectIndex(i: number, n: number): number {
  if (n === 1) return 0;
  const p = 2 * (n - 1);
  i = ((i % p) + p) % p;
  return i < n ? i : p - i;
}

/** Source index of position j of x padded with `mode`; -1 for a zero (constant) pad sample. */
export function padIndex(j: number, n: number, mode: PadMode): number {
  if (j >= 0 && j < n) return j;
  return mode === 'constant' ? -1 : reflectIndex(j, n);
}
