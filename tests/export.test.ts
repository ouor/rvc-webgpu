import { unzipSync } from 'fflate';
import { describe, expect, it } from 'vitest';
import { conformStems, stemScale } from '../src/audio/export';
import { exportApi } from '../src/audio/export-encode';
import { resample } from '../src/dsp/resample';
import type { Stereo } from '../src/shared/types';

const rs = async (x: Float32Array, a: number, b: number) => resample(x, a, b);
const st = (n: number, v: number): Stereo => ({ sr: 44100, l: new Float32Array(n).fill(v), r: new Float32Array(n).fill(-v) });

describe('export conform (I-14, QA B16)', () => {
  it('converted becomes 44.1 kHz stereo at the instrumental length', async () => {
    const inst = st(44100 * 2, 0.5), voc = st(44100 * 2, 0.2);
    const c = await conformStems({ converted: { sr: 40000, data: new Float32Array(79_200).fill(0.1) }, vocals: voc, instrumental: inst }, {}, rs);
    expect(c.frames).toBe(88200);
    for (const s of Object.values(c.stems)) { expect(s.sr).toBe(44100); expect(s.l.length).toBe(88200); expect(s.r.length).toBe(88200); }
    expect(c.stems.converted.r).toBe(c.stems.converted.l); // mono on both channels
    expect(c.stems.instrumental!.l).toBe(inst.l); // no copy when nothing changes
    expect(c.scale).toBe(1);
  });
  it('alignToSource pads to the whole source; one scale for the batch when a stem clips', async () => {
    const c = await conformStems({ converted: { sr: 44100, data: new Float32Array(44100).fill(0.1) }, instrumental: st(44100, 1.125) },
      { range: { start: 2, end: 3 }, sourceSeconds: 5, alignToSource: true }, rs);
    expect(c.frames).toBe(5 * 44100);
    expect(c.stems.instrumental!.l[2 * 44100 - 1]).toBe(0);
    expect(c.stems.instrumental!.l[2 * 44100]).toBeCloseTo(1.125);
    expect(c.scale).toBeCloseTo(0.999 / 1.125);
    expect(stemScale([st(10, 0.9)])).toBe(1);
  });
});

describe('export encode / zip (I-13)', () => {
  it('zip entries are byte-identical to encode(); scale prevents 16-bit clipping', async () => {
    const a = st(1000, 1.125), scale = 0.999 / 1.125;
    const wav = new Uint8Array(await (await exportApi.encode(a, 'pcm16', scale)).arrayBuffer());
    const z = unzipSync(new Uint8Array(await (await exportApi.zip([{ name: 'a.wav', audio: a, scale }, { name: 'settings.txt', text: 'key +2' }], 'pcm16')).arrayBuffer()));
    expect(z['a.wav']).toEqual(wav);
    expect(new TextDecoder().decode(z['settings.txt'])).toBe('key +2');
    const pcm = new Int16Array(wav.buffer, 44, 2000);
    expect(pcm[0]).toBe(Math.round(0.999 * 32767));
  });
});
