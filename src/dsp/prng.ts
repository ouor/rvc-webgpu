// Seeded PRNG for the rnd / noise graph inputs (ARCHITECTURE §0: no Random ops, reproducible runs).

export interface Prng { next(): number; gaussian(n: number): Float32Array }

/** mulberry32 uniform [0, 1) + Box-Muller standard normal. */
export function createPrng(seed: number): Prng {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    gaussian(n) {
      const out = new Float32Array(n);
      for (let i = 0; i < n; i += 2) {
        const r = Math.sqrt(-2 * Math.log(1 - next()));
        const v = 2 * Math.PI * next();
        out[i] = r * Math.cos(v);
        if (i + 1 < n) out[i + 1] = r * Math.sin(v);
      }
      return out;
    },
  };
}
