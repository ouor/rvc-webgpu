// S4 result (DESIGN 9.9, 9.11, 9.12): A/B on top, player + mixer (S/M), files by purpose, rerun panel (F2).
import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, CircleCheck, Download as DownloadIcon, Info, Loader2, Recycle, Share2, TriangleAlert, X } from 'lucide-react';
import { Button } from './ui/button';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { HelpTip, SliderField } from './Field';
import { OptionsFields } from './Options';
import { AudioPlayer, PlayButton, Waveform, bufferPeaks, seek, toggle, usePlayer } from './Player';
import { DEFAULT_MIX } from '@/audio/mixer';
import type { AudioFormat } from '@/audio/wav';
import { abLabel, changedKeys, effectiveMix, estimateRun, fmtBytes, fmtEta, fmtSigned, fmtTime, formatChanges, type RunStage, type StemKind } from '../lib/format';
import { useShallow } from 'zustand/react/shallow';
import { cn } from '../lib/utils';
import { MP3_BITRATE, backToWork, exportStem, fileName, setAB, setFormat, setMix, shareMix, start, toggleMute, toggleSolo } from '../controller';
import { selBusy, selEngineBusy, selCurrentResult, useApp, type Options, type Result } from '../store';
import { useT, type T } from '../i18n';

const AB_TIP = 'web-rvc:abTipSeen';

/** "쓴 설정 보기": every option with its original term (DESIGN 9.3 result subtitle). */
const allSettings = (t: T, o: Options, r: Result) => [
  [t('opt.key'), `${fmtSigned(o.key)} ${t('opt.semitone')}`],
  r.hasIndex && [t('opt.indexRate'), o.indexRate],
  [t('opt.protect'), o.protect],
  [t('opt.rms'), o.rmsMixRate],
  r.mode === 'song' && [t('opt.agg'), o.aggressiveness],
  o.sid > 0 && [t('opt.sid'), o.sid],
  o.range && [t('opt.secRange'), `${fmtTime(o.range.start)}–${fmtTime(o.range.end)}`],
].filter(Boolean) as [string, string | number][];

/** Bytes of one exported file in a format (44.1 kHz stereo; MP3 at the fixed CBR rate). */
const fileBytes = (seconds: number, f: AudioFormat) => (f === 'mp3' ? (seconds * MP3_BITRATE) / 8 : seconds * 44100 * 2 * (f === 'f32' ? 4 : 2));

function ABSwitch({ result, prev }: { result: Result; prev: Result }) {
  const t = useT();
  const ab = useApp((s) => s.ab);
  const [tip, setTip] = useState(() => { try { return !localStorage.getItem(AB_TIP); } catch { return false; } });
  const closeTip = () => { setTip(false); try { localStorage.setItem(AB_TIP, '1'); } catch { /* ignore */ } };
  const items = [
    { v: 'A' as const, label: t('res.abA', { d: abLabel(t, prev.opts, result.opts) }) },
    { v: 'B' as const, label: t('res.abB', { d: abLabel(t, result.opts, prev.opts) }) },
  ];
  return (
    <div className="relative mb-4">
      <div role="radiogroup" aria-label="A/B" className="grid grid-cols-2 p-1 rounded-xl bg-muted"
        onKeyDown={(e) => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); setAB(ab === 'A' ? 'B' : 'A'); } }}>
        {items.map((it) => (
          <button key={it.v} type="button" role="radio" aria-checked={ab === it.v} tabIndex={ab === it.v ? 0 : -1}
            onClick={() => { setAB(it.v); closeTip(); }}
            className={cn('h-11 rounded-lg text-small font-semibold tnum transition-colors focus-visible:outline-2 focus-visible:outline-ring',
              ab === it.v ? 'bg-card text-foreground shadow-card' : 'text-muted-foreground hover:text-foreground')}>
            {it.label}
          </button>
        ))}
      </div>
      <p className="text-caption text-muted-foreground mt-1.5 text-right">{t('res.abHint')} <kbd className="px-1 rounded-sm bg-muted">A</kbd> <kbd className="px-1 rounded-sm bg-muted">B</kbd></p>
      {tip && (
        <div role="status" className="absolute left-1/2 -translate-x-1/2 -top-11 z-10 flex items-center gap-2 rounded-lg bg-popover text-popover-foreground border shadow-pop pl-3 pr-1 py-1 text-small whitespace-nowrap">
          {t('res.abTip')}
          <button type="button" className="grid place-items-center size-6 rounded-md hover:bg-muted" aria-label={t('common.close')} onClick={closeTip}><X className="size-3.5" /></button>
        </div>
      )}
    </div>
  );
}

/** S/M toggles (DESIGN 9.12). */
function SoloMute({ k, name }: { k: 'vocal' | 'inst'; name: string }) {
  const t = useT();
  const { solo, muted } = useApp(useShallow((s) => ({ solo: s.solo === k, muted: s.mute[k] })));
  const base = 'grid place-items-center size-7 rounded-md text-caption font-bold';
  return (
    <span className="inline-flex gap-1 ml-1">
      <button type="button" aria-pressed={solo} aria-label={t('mix.solo', { name })} title={t('mix.solo', { name })} onClick={() => toggleSolo(k)}
        className={cn(base, solo ? 'bg-warning text-warning-foreground' : 'bg-muted text-muted-foreground hover:text-foreground')}>S</button>
      <button type="button" aria-pressed={muted} aria-label={t('mix.mute', { name })} title={t('mix.mute', { name })} onClick={() => toggleMute(k)}
        className={cn(base, muted ? 'bg-destructive text-destructive-foreground' : 'bg-muted text-muted-foreground hover:text-foreground')}>M</button>
    </span>
  );
}

function Hero({ result, cur }: { result: Result; cur: Result }) {
  const t = useT();
  const { prev, mix, mute, solo } = useApp(useShallow((s) => ({ prev: s.prev, mix: s.mix, mute: s.mute, solo: s.solo })));
  const stems = useMemo(() => ({ vocal: cur.converted, inst: cur.inst }), [cur]);
  const live = useMemo(() => effectiveMix(mix, mute, solo), [mix, mute, solo]);
  const player = usePlayer(stems, live);
  const song = cur.mode === 'song';
  const busy = useApp(selBusy);
  const title = useRef<HTMLHeadingElement>(null);
  useEffect(() => { title.current?.focus(); }, [result.seq]); // QA B9: focus lands on the new result

  // DESIGN 6 shortcuts: Space play, arrows +-5 s, A/B compare (not while typing or on a control).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input,textarea,select,button,[role=slider],[role=radio],[role=combobox],[contenteditable],[role=dialog]')) return;
      if (e.key === ' ') { e.preventDefault(); toggle(player); }
      else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); seek(player, player.position + (e.key === 'ArrowRight' ? 5 : -5)); }
      else if ((e.key === 'a' || e.key === 'A') && prev) setAB('A');
      else if ((e.key === 'b' || e.key === 'B') && prev) setAB('B');
    };
    addEventListener('keydown', onKey);
    return () => removeEventListener('keydown', onKey);
  }, [player, prev]);

  const gain = (k: 'vocal' | 'inst') => {
    const key = k === 'vocal' ? 'vocalGainDb' : 'instGainDb';
    const name = k === 'vocal' ? t('mix.vocal') : t('mix.inst');
    return (
      <div className={cn(mute[k] && '[&_[data-slot=slider]]:opacity-40 [&_input]:opacity-40')}>
        <SliderField label={name} name={name} dot={k === 'vocal' ? 'bg-stem-vocal' : 'bg-stem-inst'} badge={<SoloMute k={k} name={name} />}
          value={mix[key]} onChange={(v) => setMix({ [key]: v })} min={-24} max={6} step={0.5} unit="dB" defaultValue={DEFAULT_MIX[key]} valueText={(v) => `${v} dB`} />
      </div>
    );
  };

  return (
    <section className={cn('rounded-2xl bg-card shadow-card p-6', busy && 'opacity-55')} inert={busy}>
      <p className="flex items-center gap-1.5 text-small font-semibold text-success mb-3">
        <CircleCheck className="size-4" />{song ? t('res.doneSong', { t: fmtTime(result.ms / 1000) }) : t('res.doneVoice', { t: fmtTime(result.ms / 1000) })}
      </p>
      {prev && <ABSwitch result={result} prev={prev} />}
      <h1 ref={title} tabIndex={-1} className="text-display break-words focus:outline-none">{cur.songName}</h1>
      <p className="text-small text-muted-foreground mt-1 tnum flex items-center gap-1.5 flex-wrap">
        <span>{t('opt.keyShort')} {fmtSigned(cur.opts.key)} · {cur.modelName} ·</span>
        <Popover>
          <PopoverTrigger asChild><button type="button" className="font-semibold text-accent-foreground hover:underline underline-offset-2">{t('res.settingsUsed')}</button></PopoverTrigger>
          <PopoverContent align="start" className="w-80">
            <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1.5 text-small tnum">
              {allSettings(t, cur.opts, cur).map(([k, v]) => <div key={k} className="contents"><dt className="text-muted-foreground">{k}</dt><dd className="font-semibold text-right">{v}</dd></div>)}
            </dl>
          </PopoverContent>
        </Popover>
      </p>
      <div className="mt-5">
        {/* The mix waveform is drawn from the instrumental (song) as an approximation; rendering the real mix for peaks is not worth it. */}
        <AudioPlayer player={player} peaks={bufferPeaks(cur.inst ?? cur.converted, 1600)} duration={player.duration} color={song ? 'stem-mix' : 'stem-vocal'} height={80} size="lg" label={song ? t('stem.mix') : t('stem.converted')} />
      </div>
      {song && <>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-5">
          {gain('vocal')}
          {gain('inst')}
          <SliderField label={t('mix.reverb')} help={t('mix.reverbHelp')} value={Math.round(mix.reverb * 100)} onChange={(v) => setMix({ reverb: v / 100 })} min={0} max={100} step={1} unit="%" defaultValue={0} valueText={(v) => `${v}%`} />
        </div>
        <p className="text-caption text-muted-foreground mt-3">{t('mix.note')}<br />{t('mix.note2')}</p>
      </>}
    </section>
  );
}

const BAR: Record<string, string> = { 'stem-mix': 'bg-stem-mix', 'stem-vocal': 'bg-stem-vocal', 'stem-orig': 'bg-stem-orig', 'stem-inst': 'bg-stem-inst' };
const canShareFiles = () => { try { return !!navigator.canShare?.({ files: [new File([''], 'x.mp3', { type: 'audio/mpeg' })] }); } catch { return false; } };

function StemRow({ cur, kind, buffer, color, title, primary, tag }: { cur: Result; kind: StemKind; buffer: AudioBuffer; color: string; title: string; primary?: boolean; tag?: string }) {
  const t = useT();
  const { format, mix, mute } = useApp(useShallow((s) => ({ format: s.format, mix: s.mix, mute: s.mute })));
  const live = useMemo(() => effectiveMix(mix, mute), [mix, mute]);
  // The mix row plays the mix as it will be saved (mute, no solo); other rows play their stem alone.
  const stems = useMemo(() => (kind === 'mix' ? { vocal: cur.converted, inst: cur.inst } : { vocal: buffer }), [kind, cur, buffer]);
  const player = usePlayer(stems, kind === 'mix' ? live : undefined);
  const [busy, setBusy] = useState<'save' | 'share'>();
  const share = primary && canShareFiles();
  const name = fileName(cur, kind);
  const off = kind === 'mix' && (mute.vocal ? t('mix.mutedVocal') : mute.inst ? t('mix.mutedInst') : undefined);
  const run = async (what: 'save' | 'share') => { setBusy(what); try { await (what === 'save' ? exportStem(kind, cur) : shareMix(cur)); } finally { setBusy(undefined); } };
  return (
    <li className="flex items-center gap-3 py-3">
      <PlayButton player={player} size="sm" label={title} />
      <span className={cn('w-1 self-stretch rounded-full', BAR[color])} />
      <div className="w-56 min-w-0 shrink-0">
        <p className="font-semibold text-small flex items-center gap-1.5 flex-wrap">{title}
          {tag && <span className="px-1.5 py-0.5 rounded-sm bg-accent text-accent-foreground text-caption">{tag}</span>}
          {off && <span className="px-1.5 py-0.5 rounded-sm bg-warning-soft text-warning text-caption">{off}</span>}
        </p>
        <p className="text-caption text-muted-foreground truncate" title={name}>{name}</p>
      </div>
      <div className="flex-1 min-w-0 hidden md:block"><Waveform peaks={bufferPeaks(buffer, 400)} duration={buffer.duration} color={color} height={32} /></div>
      <span className="text-caption text-muted-foreground tnum w-14 text-right">{fmtBytes(fileBytes(buffer.duration, format))}</span>
      {share && (
        <Button variant="secondary" size="icon-sm" aria-label={t('res.share')} title={t('res.share')} disabled={!!busy} onClick={() => void run('share')}>
          {busy === 'share' ? <Loader2 className="animate-spin" /> : <Share2 />}
        </Button>
      )}
      <Button variant={primary ? 'default' : 'secondary'} size="sm" className="min-w-20" disabled={!!busy} onClick={() => void run('save')}>
        {busy === 'save' ? <Loader2 className="animate-spin" /> : <DownloadIcon />}{t('common.save')}
      </Button>
    </li>
  );
}

const FORMATS: { f: AudioFormat; name: string; use: 'res.fmtMp3' | 'res.fmtWav' | 'res.fmtF32' }[] = [
  { f: 'mp3', name: 'MP3', use: 'res.fmtMp3' }, { f: 'pcm16', name: 'WAV', use: 'res.fmtWav' }, { f: 'f32', name: 'WAV 32-bit float', use: 'res.fmtF32' },
];

/** DESIGN 9.11: three format cards by purpose with the mix size for this song (radio group, arrows select). */
function FormatPicker({ seconds, voice }: { seconds: number; voice: boolean }) {
  const t = useT();
  const format = useApp((s) => s.format);
  const move = (d: number) => setFormat(FORMATS[(FORMATS.findIndex((x) => x.f === format) + d + FORMATS.length) % FORMATS.length].f);
  return (
    <div role="radiogroup" aria-label="Format" className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-2"
      onKeyDown={(e) => { if (['ArrowRight', 'ArrowDown'].includes(e.key)) { e.preventDefault(); move(1); } else if (['ArrowLeft', 'ArrowUp'].includes(e.key)) { e.preventDefault(); move(-1); } }}>
      {FORMATS.map((x) => {
        const on = format === x.f;
        return (
          <button key={x.f} type="button" role="radio" aria-checked={on} tabIndex={on ? 0 : -1} onClick={() => setFormat(x.f)}
            className={cn('text-left rounded-xl border px-3 py-2 min-h-16 transition-colors focus-visible:outline-2 focus-visible:outline-ring',
              on ? 'border-2 border-primary bg-accent/50 px-[11px] py-[7px]' : 'hover:bg-muted')}>
            <span className="flex items-center gap-2 text-small font-semibold">
              <span className={cn('size-3.5 rounded-full border-2 shrink-0', on ? 'border-primary bg-primary shadow-[inset_0_0_0_2px_hsl(var(--card))]' : 'border-input')} />{x.name}
            </span>
            <span className="block text-caption text-muted-foreground mt-0.5 pl-5.5">{t(x.use)}</span>
            <span className="block text-caption text-muted-foreground tnum pl-5.5">{t(voice ? 'res.fmtSizeVoice' : 'res.fmtSize', { n: fmtBytes(fileBytes(seconds, x.f)) })}</span>
          </button>
        );
      })}
    </div>
  );
}

function Files({ cur }: { cur: Result }) {
  const t = useT();
  const song = cur.mode === 'song';
  const seconds = (cur.inst ?? cur.converted).duration;
  return (
    <section className="rounded-2xl bg-card shadow-card p-6">
      <h2 className="text-h2">{t('res.files')}</h2>
      <FormatPicker seconds={seconds} voice={!song} />
      {!cur.savedOut && (
        <p className="mt-4 flex items-start gap-2 rounded-lg bg-warning-soft px-3 py-2 text-small"><TriangleAlert className="size-4 mt-0.5 shrink-0 text-warning" />{t('res.notSaved')}</p>
      )}
      {song && <p className="mt-3 flex items-center gap-2 text-small"><Info className="size-4 shrink-0 text-accent-foreground" />{t('res.mixNote')}</p>}
      <ul className="mt-2 divide-y">
        {song && cur.inst && <StemRow cur={cur} kind="mix" buffer={cur.inst} color="stem-mix" title={t('stem.mix')} tag={t('res.recommended')} primary />}
        <StemRow cur={cur} kind="converted" buffer={cur.converted} color="stem-vocal" title={t('stem.converted')} primary={!song} />
        {song && cur.vocals && <StemRow cur={cur} kind="vocals" buffer={cur.vocals} color="stem-orig" title={t('stem.vocals')} />}
        {song && cur.inst && <StemRow cur={cur} kind="inst" buffer={cur.inst} color="stem-inst" title={t('stem.inst')} />}
      </ul>
    </section>
  );
}

function Rerun({ result: r }: { result: Result }) {
  const t = useT();
  const { opts, prev, env } = useApp(useShallow((s) => ({ opts: s.opts, prev: s.prev, env: s.env })));
  const busy = useApp(selEngineBusy);
  const changes = changedKeys(r.opts, opts).length ? formatChanges(t, r.opts, opts) : undefined;
  const sepAgain = r.mode === 'song' && opts.aggressiveness !== r.opts.aggressiveness;
  const seconds = r.opts.range ? r.opts.range.end - r.opts.range.start : r.converted.duration;
  const stages: RunStage[] = sepAgain ? ['separate', 'convert'] : ['convert'];
  const eta = fmtEta(t, estimateRun(seconds, env?.backend ?? 'webgpu', stages));
  return (
    <aside className={cn('rounded-2xl bg-card shadow-card p-6 lg:sticky lg:top-20', busy && 'opacity-55')} inert={busy}>
      <h2 className="text-h2">{t('rerun.title')}</h2>
      <p className="text-small text-muted-foreground mt-1">{r.mode === 'song' ? t('rerun.subSong') : t('rerun.subVoice')}</p>
      <div className="mt-5"><OptionsFields rerun /></div>
      {changes && (
        <div className="mt-5 rounded-xl bg-accent px-4 py-3">
          <p className="text-small font-semibold text-accent-foreground flex items-center gap-1.5"><Recycle className="size-4" />
            {sepAgain ? t('rerun.sepAgain') : <>{t('rerun.reuseSong')}<HelpTip text={r.mode === 'song' ? t('rerun.reuseSongHelp') : t('rerun.reuseVoiceHelp')} /></>}
          </p>
          <p className="text-caption text-muted-foreground mt-1 tnum">{changes}</p>
          <p className="text-caption text-muted-foreground mt-0.5">{sepAgain ? t('rerun.sepAgainSub', { eta }) : t('rerun.reuseSub', { eta })}</p>
        </div>
      )}
      <Button size="cta" className="mt-5 w-full" disabled={!changes || busy}
        aria-describedby="rerun-why" onClick={() => void start({ rerun: true })}>{t('rerun.button')}</Button>
      <p id="rerun-why" className="text-caption text-muted-foreground mt-2 text-center">{!changes ? t('rerun.noChange') : prev ? t('rerun.abReplace') : t('rerun.abKeep')}</p>
    </aside>
  );
}

export function ResultView() {
  const t = useT();
  const result = useApp((s) => s.result);
  const cur = useApp(selCurrentResult);
  if (!result || !cur) return null;
  return (
    <main className="mx-auto max-w-[1280px] px-6 pt-5">
      <div className="mb-4">
        <Button variant="ghost" size="sm" className="-ml-2" onClick={backToWork}><ArrowLeft />{t('res.back')}</Button>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-5 items-start">
        <div className="space-y-5 min-w-0"><Hero result={result} cur={cur} /><Files cur={cur} /></div>
        <Rerun result={result} />
      </div>
    </main>
  );
}
