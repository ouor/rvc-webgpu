// UI pure helpers: export file names (I-15 / MX-6) and the mute/solo mix (I-17).
import { describe, expect, it } from 'vitest';
import { effectiveMix, exportName, safe, zipName } from '../src/app/lib/format';

describe('export file names', () => {
  it('follows {song}_{kind}_{model}.{ext}', () => {
    expect(exportName('버니걸', 'rvc-kyubey', 'mix', 'mp3')).toBe('버니걸_cover_rvc-kyubey.mp3');
    expect(exportName('a', 'm', 'converted', 'wav')).toBe('a_vocals-converted_m.wav');
    expect(exportName('a', 'm', 'vocals', 'wav')).toBe('a_vocals-original_m.wav');
    expect(exportName('a', 'm', 'inst', 'wav')).toBe('a_instrumental_m.wav');
    expect(zipName('a', 'm')).toBe('a_m.zip');
  });
  it('adds the range start', () => {
    expect(exportName('a', 'm', 'mix', 'mp3', 65)).toBe('a_cover_m_from-01m05s.mp3');
    expect(exportName('a', 'm', 'mix', 'mp3', 0)).toBe('a_cover_m_from-00m00s.mp3');
  });
  it('replaces forbidden characters with "-" and keeps Korean and spaces', () => {
    expect(safe('a/b\\c:d*e?f"g<h>i|j')).toBe('a-b-c-d-e-f-g-h-i-j');
    expect(safe('노래 제목\u0001')).toBe('노래 제목-');
    expect(safe('  ')).toBe('untitled');
  });
});

describe('effectiveMix', () => {
  const m = { vocalGainDb: -3, instGainDb: 2, reverb: 0.2 };
  it('mute silences in playback and render, solo only in playback', () => {
    expect(effectiveMix(m, { vocal: false, inst: true }).instGainDb).toBe(-Infinity);
    expect(effectiveMix(m, { vocal: false, inst: false }, 'vocal')).toEqual({ ...m, instGainDb: -Infinity });
    expect(effectiveMix(m, { vocal: false, inst: false })).toEqual(m);
  });
});
