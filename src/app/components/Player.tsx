// Playback (MixPlayer on one shared AudioContext, one source at a time) and the wavesurfer waveform (draw-only, REFERENCES 3.10).
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { Pause, Play } from 'lucide-react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin, { type Region } from 'wavesurfer.js/plugins/regions';
import { MixPlayer, type MixParams, type Stems } from '@/audio/mixer';
import { peaks } from '@/audio/decode';
import { fmtTime } from '../lib/format';
import { cn } from '../lib/utils';
import { useApp } from '../store';
import { useT } from '../i18n';

let ctx: AudioContext | undefined;
const audioCtx = () => (ctx ??= new AudioContext({ sampleRate: 44100 }));

// Exclusive playback: starting one player pauses the others.
let active: MixPlayer | undefined;
const subs = new Set<() => void>();
let version = 0;
const notify = () => { version++; subs.forEach((f) => f()); };
const subscribe = (f: () => void) => { subs.add(f); return () => subs.delete(f); };

export function toggle(p: MixPlayer) {
  if (p.isPlaying) p.pause();
  else { if (active && active !== p) active.pause(); active = p; p.play(); }
  notify();
}
export function seek(p: MixPlayer, t: number) { p.seek(t); notify(); }
const onEnd = (p: MixPlayer, f?: () => void) => { p.onEnded = f; };
export const pauseAll = () => { active?.pause(); notify(); };

/** Player bound to stems; keeps the position when stems change (A/B switch at the same spot). */
export function usePlayer(stems: Stems | undefined, mix?: MixParams) {
  const player = useMemo(() => new MixPlayer(audioCtx()), []);
  useSyncExternalStore(subscribe, () => version);
  useEffect(() => {
    if (!stems) return;
    const pos = player.position, playing = player.isPlaying;
    player.load(stems);
    if (pos > 0) { if (playing) player.play(pos); else player.seek(pos); }
    notify();
  }, [player, stems]);
  useEffect(() => { if (mix) player.set(mix); }, [player, mix]);
  useEffect(() => { onEnd(player, notify); return () => { player.pause(); onEnd(player, undefined); }; }, [player]);
  return player;
}

const cssColor = (token: string, alpha = 1) => {
  const v = getComputedStyle(document.documentElement).getPropertyValue(`--${token}`).trim();
  return `hsl(${v} / ${alpha})`;
};

interface WaveformProps {
  peaks: Float32Array;
  duration: number;
  color: string; // token name, e.g. 'stem-mix'
  height: number;
  player?: MixPlayer;
  region?: { start: number; end: number };
  onRegion?: (r: { start: number; end: number } | undefined) => void;
  label?: string;
  timeRef?: React.RefObject<HTMLSpanElement | null>;
}

export function Waveform({ peaks, duration, color, height, player, region, onRegion, label, timeRef }: WaveformProps) {
  const t = useT();
  const box = useRef<HTMLDivElement>(null);
  const ws = useRef<WaveSurfer>(undefined);
  const regions = useRef<ReturnType<typeof RegionsPlugin.create>>(undefined);
  const theme = useApp((s) => s.resolvedTheme);
  const onRegionRef = useRef(onRegion);
  useEffect(() => { onRegionRef.current = onRegion; });
  const [pos, setPos] = useState(0);

  useEffect(() => {
    if (!box.current) return;
    const w = WaveSurfer.create({
      container: box.current, height, peaks: [peaks], duration, normalize: true,
      waveColor: cssColor(color, 0.35), progressColor: cssColor(color), cursorColor: cssColor('foreground'), cursorWidth: player ? 2 : 0,
      barWidth: 2, barGap: 2, barRadius: 2, interact: !!player, dragToSeek: !!player && !onRegion,
    });
    ws.current = w;
    if (player) w.on('interaction', (time) => seek(player, time));
    if (onRegion) {
      const rp = w.registerPlugin(RegionsPlugin.create());
      regions.current = rp;
      rp.enableDragSelection({ color: cssColor('primary', 0.15) });
      const emit = (r: Region) => onRegionRef.current?.({ start: +r.start.toFixed(1), end: +r.end.toFixed(1) });
      rp.on('region-created', (r) => { rp.getRegions().forEach((x) => x !== r && x.remove()); emit(r); });
      rp.on('region-updated', emit);
    }
    return () => { w.destroy(); ws.current = undefined; regions.current = undefined; };
  }, [peaks, duration, color, height, player, !!onRegion, theme]); // eslint-disable-line react-hooks/exhaustive-deps

  // External range edits (number inputs) -> region handles.
  useEffect(() => {
    const rp = regions.current;
    if (!rp) return;
    const cur = rp.getRegions()[0];
    if (!region) { rp.clearRegions(); return; }
    if (cur) { if (Math.abs(cur.start - region.start) > 0.05 || Math.abs(cur.end - region.end) > 0.05) cur.setOptions({ start: region.start, end: region.end }); }
    else rp.addRegion({ start: region.start, end: region.end, color: cssColor('primary', 0.15) });
  }, [region, peaks, theme]);

  // rAF cursor while playing (not React state per frame).
  const playing = player?.isPlaying;
  const v = useSyncExternalStore(subscribe, () => version); // paused seeks change only the position
  useEffect(() => {
    if (!player) return;
    let raf = 0;
    const tick = () => {
      ws.current?.setTime(player.position);
      if (timeRef?.current) timeRef.current.textContent = fmtTime(player.position);
      if (player.isPlaying) raf = requestAnimationFrame(tick);
      else setPos(player.position);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [player, playing, timeRef, v]);

  const onKey = (e: React.KeyboardEvent) => {
    if (!player) return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); e.stopPropagation(); seek(player, player.position + (e.key === 'ArrowRight' ? 5 : -5)); }
    if (e.key === 'Home') { e.preventDefault(); seek(player, 0); }
  };

  return (
    <div
      ref={box}
      className="relative w-full rounded-md focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
      style={{ height }}
      {...(player ? { role: 'slider', tabIndex: 0, 'aria-label': label ?? t('player.position'), 'aria-valuemin': 0, 'aria-valuemax': Math.round(duration), 'aria-valuenow': Math.round(pos), 'aria-valuetext': fmtTime(pos), onKeyDown: onKey } : { 'aria-hidden': true })}
    />
  );
}

export function PlayButton({ player, size = 'md', label, className }: { player: MixPlayer; size?: 'sm' | 'md' | 'lg'; label: string; className?: string }) {
  const t = useT();
  useSyncExternalStore(subscribe, () => version);
  const on = player.isPlaying;
  const Icon = on ? Pause : Play;
  return (
    <button type="button" onClick={() => toggle(player)} aria-label={`${label} ${on ? t('player.pause') : t('player.play')}`} aria-pressed={on}
      className={cn('grid place-items-center rounded-full bg-foreground text-background shrink-0 hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring',
        size === 'lg' ? 'size-14' : size === 'md' ? 'size-9' : 'size-8', className)}>
      <Icon className={cn('fill-current', size === 'lg' ? 'size-6' : 'size-4')} />
    </button>
  );
}

/** AudioPlayer: play button + waveform + times. */
export function AudioPlayer({ player, peaks, duration, color, height, size, label, region, onRegion }: {
  player: MixPlayer; peaks: Float32Array; duration: number; color: string; height: number; size: 'md' | 'lg'; label: string;
  region?: { start: number; end: number }; onRegion?: (r: { start: number; end: number } | undefined) => void;
}) {
  const timeRef = useRef<HTMLSpanElement>(null);
  return (
    <div>
      <div className="flex items-center gap-3">
        <PlayButton player={player} size={size} label={label} />
        <div className="flex-1 min-w-0">
          <Waveform peaks={peaks} duration={duration} color={color} height={height} player={player} region={region} onRegion={onRegion} label={label} timeRef={timeRef} />
        </div>
      </div>
      <div className={cn('flex justify-between text-caption text-muted-foreground tnum mt-1', size === 'lg' ? 'pl-[68px]' : 'pl-12')}>
        <span ref={timeRef}>0:00</span><span>{fmtTime(duration)}</span>
      </div>
    </div>
  );
}

/** Peaks of an AudioBuffer's first channel, memoised by buffer identity. */
const peakCache = new WeakMap<AudioBuffer, Float32Array>();
export function bufferPeaks(b: AudioBuffer, points = 1000) {
  let p = peakCache.get(b);
  if (!p) peakCache.set(b, (p = peaks(b.getChannelData(0), points)));
  return p;
}
