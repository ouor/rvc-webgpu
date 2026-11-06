import { describe, expect, it } from 'vitest';
import { resamplePoly } from '../src/dsp/resample';
import { filtfilt, HPF48_A, HPF48_B } from '../src/dsp/filtfilt';
import { istft, stft } from '../src/dsp/stft';
import { changeRms } from '../src/dsp/rms';
import { Fft } from '../src/dsp/fft';
import { createPrng } from '../src/dsp/prng';
import { hasFixture, loadFixture, relErr } from './helpers';

const fx = (name: string) => (hasFixture(name) ? it : it.skip);

describe('fft', () => {
  it.each([8, 320, 640, 960, 1024])('matches a naive DFT, n=%i', (n) => {
    const x = Float64Array.from({ length: 2 * n }, (_, i) => Math.sin(i * 0.37) + (i % 7) * 0.1);
    const out = new Float64Array(2 * n);
    new Fft(n).forward(x, out);
    for (const k of [0, 1, n >> 1, n - 1]) {
      let re = 0, im = 0;
      for (let j = 0; j < n; j++) {
        const a = (-2 * Math.PI * j * k) / n;
        re += x[2 * j] * Math.cos(a) - x[2 * j + 1] * Math.sin(a);
        im += x[2 * j] * Math.sin(a) + x[2 * j + 1] * Math.cos(a);
      }
      expect(Math.abs(out[2 * k] - re) + Math.abs(out[2 * k + 1] - im)).toBeLessThan(1e-8 * n);
    }
    const back = new Float64Array(2 * n);
    new Fft(n).inverse(out, back);
    expect(relErr(back, x)).toBeLessThan(1e-10);
  });
});

describe('resample_poly', () => {
  for (const [a, b] of [[44100, 16000], [44100, 14700], [14700, 7350], [7350, 14700], [14700, 44100], [40000, 44100]]) {
    fx(`dsp-resample-${a}-${b}`)(`${a} -> ${b}`, () => {
      const f = loadFixture(`dsp-resample-${a}-${b}`);
      const y = resamplePoly(f.inputs.x as Float32Array, b, a);
      expect(y.length).toBe(f.outputs.y.length);
      expect(relErr(y, f.outputs.y as Float32Array)).toBeLessThan(1e-5);
    });
  }
});

describe('filtfilt', () => {
  fx('dsp-filtfilt-hpf48')('hpf48 matches scipy', () => {
    const f = loadFixture('dsp-filtfilt-hpf48');
    if (f.meta.b) expect(relErr(HPF48_B, f.meta.b as number[])).toBeLessThan(1e-12);
    expect(relErr(filtfilt(HPF48_B, HPF48_A, f.inputs.x as Float32Array), f.outputs.y as Float64Array)).toBeLessThan(1e-5);
  });
});

describe('stft', () => {
  for (const n of [1024, 960, 640, 512, 320]) {
    fx(`dsp-stft-${n}`)(`n_fft ${n} stft/istft match librosa`, () => {
      const f = loadFixture(`dsp-stft-${n}`);
      const s = stft(f.inputs.x as Float32Array, n, f.meta.hop as number);
      expect([s.bins, s.frames]).toEqual(f.shapes.re);
      expect(relErr(s.re, f.outputs.re as Float32Array)).toBeLessThan(1e-5);
      expect(relErr(s.im, f.outputs.im as Float32Array)).toBeLessThan(1e-5);
      const y = istft(s, f.meta.hop as number);
      expect(y.length).toBe(f.outputs.y.length);
      expect(relErr(y, f.outputs.y as Float32Array)).toBeLessThan(1e-5);
    });
  }
});

describe('change_rms', () => {
  fx('dsp-rms')('matches rvc-nano', () => {
    const f = loadFixture('dsp-rms');
    const y = changeRms(f.inputs.data1 as Float64Array, 16000, (f.inputs.data2 as Float32Array).slice(), 40000, 0.25);
    expect(relErr(y, f.outputs.y as Float32Array)).toBeLessThan(1e-5);
  });
});

describe('prng', () => {
  it('is reproducible and roughly standard normal', () => {
    const a = createPrng(42).gaussian(100001), b = createPrng(42).gaussian(100001);
    expect(a).toEqual(b);
    const mean = a.reduce((s, v) => s + v, 0) / a.length;
    const varr = a.reduce((s, v) => s + (v - mean) ** 2, 0) / a.length;
    expect(Math.abs(mean)).toBeLessThan(0.01);
    expect(Math.abs(varr - 1)).toBeLessThan(0.02);
  });
});
