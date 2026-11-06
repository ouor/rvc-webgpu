// PCM16 WAV as a chunk stream (recent jobs, ARCHITECTURE 18.2) and its reader. Pure: runs in the engine worker.

const BLOCK = 1 << 16; // frames per chunk

function header(nc: number, sr: number, frames: number): Uint8Array {
  const data = frames * nc * 2, b = new Uint8Array(44), v = new DataView(b.buffer);
  const str = (o: number, s: string) => { for (let i = 0; i < s.length; i++) b[o + i] = s.charCodeAt(i); };
  str(0, 'RIFF'); v.setUint32(4, 36 + data, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, nc, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * nc * 2, true); v.setUint16(32, nc * 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, data, true);
  return b;
}

/** 16-bit WAV of `channels` (same length), each sample multiplied by `scale` first; no single big buffer. */
export function* wavChunks(channels: Float32Array[], sr: number, scale = 1): Generator<Uint8Array> {
  const nc = channels.length, n = channels[0].length;
  yield header(nc, sr, n);
  for (let s = 0; s < n; s += BLOCK) {
    const len = Math.min(BLOCK, n - s), out = new Int16Array(len * nc);
    for (let i = 0; i < len; i++) for (let c = 0; c < nc; c++) {
      out[i * nc + c] = Math.max(-32768, Math.min(32767, Math.round(channels[c][s + i] * scale * 32767)));
    }
    yield new Uint8Array(out.buffer);
  }
}

export const wavBytes = (channels: number, frames: number) => 44 + channels * frames * 2;

/** Reads a wavChunks() file back, divided by `scale` (original level). */
export async function readWav16(b: Blob, scale = 1): Promise<{ sr: number; channels: Float32Array[] }> {
  const buf = await b.arrayBuffer(), v = new DataView(buf);
  if (buf.byteLength < 44 || v.getUint32(0, false) !== 0x52494646 || v.getUint16(34, true) !== 16) throw new Error('not a 16-bit wav');
  const nc = v.getUint16(22, true), sr = v.getUint32(24, true);
  const n = Math.floor(Math.min(v.getUint32(40, true), buf.byteLength - 44) / (2 * nc));
  const pcm = new Int16Array(buf, 44, n * nc), k = 1 / (32767 * scale);
  const channels = Array.from({ length: nc }, () => new Float32Array(n));
  for (let i = 0; i < n; i++) for (let c = 0; c < nc; c++) channels[c][i] = pcm[i * nc + c] * k;
  return { sr, channels };
}
