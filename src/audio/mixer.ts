// Mix graph shared by live playback (MixPlayer) and the final render (renderMix): AC-10 by construction.
//   vocal -> vocalGain -> master            inst -> instGain -> master
//                 \-> convolver -> wetGain -/
import { createReverbIR } from './reverb';

export interface MixParams { vocalGainDb: number; instGainDb: number; reverb: number /* 0..1 */ }
export interface Stems { vocal: AudioBuffer; inst?: AudioBuffer }
export const DEFAULT_MIX: MixParams = { vocalGainDb: 0, instGainDb: 0, reverb: 0 };

const db = (x: number) => 10 ** (x / 20);
const PEAK = db(-1); // MX-3: -1 dBFS ceiling for the rendered mix

interface Graph { vocalGain: GainNode; instGain: GainNode; wetGain: GainNode; master: GainNode }

function buildGraph(ctx: BaseAudioContext, ir: AudioBuffer): Graph {
  const master = ctx.createGain();
  const vocalGain = ctx.createGain(), instGain = ctx.createGain(), wetGain = ctx.createGain();
  const conv = ctx.createConvolver();
  conv.buffer = ir;
  vocalGain.connect(master);
  vocalGain.connect(conv).connect(wetGain).connect(master);
  instGain.connect(master);
  master.connect(ctx.destination);
  return { vocalGain, instGain, wetGain, master };
}

function apply(g: Graph, p: MixParams, at?: number) {
  const set = (n: GainNode, v: number) => (at === undefined ? (n.gain.value = v) : n.gain.setTargetAtTime(v, at, 0.03));
  set(g.vocalGain, db(p.vocalGainDb));
  set(g.instGain, db(p.instGainDb));
  set(g.wetGain, p.reverb * 0.6); // 100 % reverb = wet at -4.4 dB under the dry vocal
}

/** Mix length: the instrumental (= original song), or the vocal alone in voice mode (AC-1). */
const mixLength = (s: Stems, sr: number) => (s.inst ? s.inst.length : Math.ceil(s.vocal.duration * sr));

export class MixPlayer {
  private graph?: Graph;
  private sources: AudioBufferSourceNode[] = [];
  private stems?: Stems;
  private params: MixParams = DEFAULT_MIX;
  private startedAt = 0;
  private offset = 0;
  private playing = false;
  onEnded?: () => void;

  constructor(private ctx: AudioContext) {}

  load(stems: Stems): void {
    this.pause();
    this.offset = 0;
    this.stems = stems;
    this.graph ??= buildGraph(this.ctx, createReverbIR(this.ctx));
    apply(this.graph, this.params);
  }

  set(p: MixParams): void {
    this.params = p;
    if (this.graph) apply(this.graph, p, this.ctx.currentTime); // ~30 ms time constant, well under 0.2 s
  }

  get duration(): number { return this.stems ? mixLength(this.stems, this.ctx.sampleRate) / this.ctx.sampleRate : 0; }
  get position(): number { return this.playing ? this.offset + this.ctx.currentTime - this.startedAt : this.offset; }
  get isPlaying(): boolean { return this.playing; }

  play(offset = this.position): void {
    if (!this.stems || !this.graph) return;
    this.stopSources();
    void this.ctx.resume();
    this.offset = Math.max(0, Math.min(offset, this.duration));
    this.startedAt = this.ctx.currentTime;
    const start = (buf: AudioBuffer, to: GainNode) => {
      const s = this.ctx.createBufferSource();
      s.buffer = buf;
      s.connect(to);
      s.start(0, this.offset);
      this.sources.push(s);
      return s;
    };
    const main = start(this.stems.inst ?? this.stems.vocal, this.stems.inst ? this.graph.instGain : this.graph.vocalGain);
    if (this.stems.inst) start(this.stems.vocal, this.graph.vocalGain);
    main.onended = () => { if (this.sources.includes(main)) { this.playing = false; this.offset = 0; this.sources = []; this.onEnded?.(); } };
    this.playing = true;
  }

  /** Moves the playhead; keeps playing if playing, otherwise the next play() starts here. */
  seek(t: number): void {
    if (this.playing) this.play(t);
    else this.offset = Math.max(0, Math.min(t, this.duration));
  }

  pause(): void {
    if (!this.playing) return;
    this.offset = this.position;
    this.stopSources();
    this.playing = false;
  }

  private stopSources() {
    const s = this.sources;
    this.sources = [];
    for (const x of s) { x.onended = null; try { x.stop(); } catch { /* not started */ } x.disconnect(); }
  }
}

/** Offline render at 44.1 kHz stereo with the same graph, then scaled so the peak is at most -1 dBFS. */
export async function renderMix(stems: Stems, p: MixParams, sampleRate = 44100): Promise<AudioBuffer> {
  const ctx = new OfflineAudioContext(2, mixLength(stems, sampleRate), sampleRate);
  const g = buildGraph(ctx, createReverbIR(ctx));
  apply(g, p);
  const play = (b: AudioBuffer, to: GainNode) => { const s = ctx.createBufferSource(); s.buffer = b; s.connect(to); s.start(0); };
  play(stems.vocal, g.vocalGain);
  if (stems.inst) play(stems.inst, g.instGain);
  const out = await ctx.startRendering();
  let peak = 0;
  for (let c = 0; c < out.numberOfChannels; c++) for (const x of out.getChannelData(c)) peak = Math.max(peak, Math.abs(x));
  if (peak > PEAK) for (let c = 0; c < out.numberOfChannels; c++) { const d = out.getChannelData(c); for (let i = 0; i < d.length; i++) d[i] *= PEAK / peak; }
  return out;
}
