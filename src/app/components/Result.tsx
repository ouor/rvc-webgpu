// S4 result: player + mixer, A/B, stem downloads, rerun panel (F2).
import { useEffect, useMemo, useState } from 'react';
import { ArrowLeft, CircleCheck, Download as DownloadIcon, Info, Loader2, Recycle } from 'lucide-react';
import { Button } from './ui/button';
import { SliderField, Segmented } from './Field';
import { OptionsFields } from './Options';
import { AudioPlayer, PlayButton, Waveform, bufferPeaks, seek, toggle, usePlayer } from './Player';
import { DEFAULT_MIX } from '@/audio/mixer';
import { changedKeys, estimateRun, fmtBytes, fmtEta, fmtSigned, fmtTime, formatChanges, type RunStage } from '../lib/format';
import { useShallow } from 'zustand/react/shallow';
import { cn } from '../lib/utils';
import { backToWork, exportStem, fileName, setAB, setFormat, setMix, start, type StemKind } from '../controller';
import { selBusy, selCurrentResult, useApp, type Options, type Result } from '../store';
import { useT, type T } from '../i18n';

const summary = (t: T, o: Options, r: Result) =>
  [`${t('opt.keyShort')} ${fmtSigned(o.key)}`, r.hasIndex && `Index ${o.indexRate}`, `Protect ${o.protect}`, r.mode === 'song' && `${t('opt.aggShort')} ${o.aggressiveness}`, o.range && `${fmtTime(o.range.start)}–${fmtTime(o.range.end)}`]
    .filter(Boolean).join(' · ');

function Hero({ result, cur }: { result: Result; cur: Result }) {
  const t = useT();
  const { prev, ab, mix } = useApp(useShallow((s) => ({ prev: s.prev, ab: s.ab, mix: s.mix })));
  const stems = useMemo(() => ({ vocal: cur.converted, inst: cur.inst }), [cur]);
  const player = usePlayer(stems, mix);
  const song = cur.mode === 'song';
  const busy = useApp(selBusy);

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

  return (
    <section className={cn('rounded-2xl bg-card shadow-card p-6', busy && 'opacity-55')} inert={busy}>
      <div className="flex items-start gap-4 flex-wrap">
        <div className="flex-1 min-w-0">
          <p className="flex items-center gap-1.5 text-small font-semibold text-success">
            <CircleCheck className="size-4" />{song ? t('res.doneSong', { t: fmtTime(result.ms / 1000) }) : t('res.doneVoice', { t: fmtTime(result.ms / 1000) })}
          </p>
          <h1 className="text-display mt-1 break-words">{fileName(cur, cur.mode === 'song' ? 'mix' : 'converted').replace(/\.\w+$/, '')}</h1>
          <p className="text-small text-muted-foreground mt-1 tnum">{prev ? `${ab}: ` : ''}{summary(t, cur.opts, cur)}</p>
        </div>
        {prev && (
          <div className="text-right">
            <Segmented label={t('res.ab')} value={ab} onChange={setAB}
              items={[{ value: 'A', label: <><b className="mr-1">A</b>{t('res.abPrev', { k: fmtSigned(prev.opts.key) })}</> }, { value: 'B', label: <><b className="mr-1">B</b>{t('res.abNow', { k: fmtSigned(result.opts.key) })}</> }]} />
            <p className="text-caption text-muted-foreground mt-1.5">{t('res.abHint')} <kbd className="px-1 rounded-sm bg-muted">A</kbd> <kbd className="px-1 rounded-sm bg-muted">B</kbd></p>
          </div>
        )}
      </div>
      <div className="mt-5">
        <AudioPlayer player={player} peaks={bufferPeaks(cur.inst ?? cur.converted, 1600)} duration={player.duration} color={song ? 'stem-mix' : 'stem-vocal'} height={80} size="lg" label={song ? t('stem.mix') : t('stem.converted')} />
      </div>
      {song && <>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-5">
          <SliderField label={t('mix.vocal')} dot="bg-stem-vocal" value={mix.vocalGainDb} onChange={(v) => setMix({ vocalGainDb: v })} min={-20} max={12} step={0.5} unit="dB" defaultValue={DEFAULT_MIX.vocalGainDb} valueText={(v) => `${v} dB`} />
          <SliderField label={t('mix.inst')} dot="bg-stem-inst" value={mix.instGainDb} onChange={(v) => setMix({ instGainDb: v })} min={-20} max={12} step={0.5} unit="dB" defaultValue={DEFAULT_MIX.instGainDb} valueText={(v) => `${v} dB`} />
          <SliderField label={t('mix.reverb')} help={t('mix.reverbHelp')} value={Math.round(mix.reverb * 100)} onChange={(v) => setMix({ reverb: v / 100 })} min={0} max={100} step={1} unit="%" defaultValue={0} valueText={(v) => `${v}%`} />
        </div>
        <p className="text-caption text-muted-foreground mt-3">{t('mix.note')}</p>
      </>}
    </section>
  );
}

const BAR: Record<string, string> = { 'stem-mix': 'bg-stem-mix', 'stem-vocal': 'bg-stem-vocal', 'stem-orig': 'bg-stem-orig', 'stem-inst': 'bg-stem-inst' };

function StemRow({ cur, kind, buffer, color, title, primary, tag }: { cur: Result; kind: StemKind; buffer: AudioBuffer; color: string; title: string; primary?: boolean; tag?: string }) {
  const t = useT();
  const format = useApp((s) => s.format);
  const mix = useApp((s) => s.mix);
  // The mix row plays the live mix graph; other rows play their stem alone.
  const stems = useMemo(() => (kind === 'mix' ? { vocal: cur.converted, inst: cur.inst } : { vocal: buffer }), [kind, cur, buffer]);
  const player = usePlayer(stems, kind === 'mix' ? mix : undefined);
  const [busy, setBusy] = useState(false);
  const ch = kind === 'converted' ? buffer.numberOfChannels : 2;
  const bytes = format === 'mp3' ? (buffer.duration * 192_000) / 8 : buffer.duration * 44100 * ch * (format === 'f32' ? 4 : 2);
  const name = fileName(cur, kind);
  return (
    <li className="flex items-center gap-3 py-3">
      <PlayButton player={player} size="sm" label={title} />
      <span className={cn('w-1 self-stretch rounded-full', BAR[color])} />
      <div className="w-56 min-w-0 shrink-0">
        <p className="font-semibold text-small flex items-center gap-1.5">{title}{tag && <span className="px-1.5 py-0.5 rounded-sm bg-accent text-accent-foreground text-caption">{tag}</span>}</p>
        <p className="text-caption text-muted-foreground truncate" title={name}>{name}</p>
      </div>
      <div className="flex-1 min-w-0 hidden md:block"><Waveform peaks={bufferPeaks(buffer, 400)} duration={buffer.duration} color={color} height={32} /></div>
      <span className="text-caption text-muted-foreground tnum w-14 text-right">{fmtBytes(bytes)}</span>
      <Button variant={primary ? 'default' : 'secondary'} size="sm" className="min-w-20" disabled={busy}
        onClick={async () => { setBusy(true); try { await exportStem(kind, cur); } finally { setBusy(false); } }}>
        {busy ? <Loader2 className="animate-spin" /> : <DownloadIcon />}{t('common.save')}
      </Button>
    </li>
  );
}

function Files({ cur }: { cur: Result }) {
  const t = useT();
  const format = useApp((s) => s.format);
  const song = cur.mode === 'song';
  return (
    <section className="rounded-2xl bg-card shadow-card p-6">
      <div className="flex items-center gap-3 flex-wrap">
        <h2 className="text-h2 flex-1">{t('res.files')}</h2>
        <span className="text-small text-muted-foreground">{t('res.format')}</span>
        <Segmented label={t('res.format')} value={format} onChange={setFormat}
          items={[{ value: 'pcm16', label: 'WAV 16-bit' }, { value: 'f32', label: 'WAV 32-bit float' }, { value: 'mp3', label: 'MP3' }]} />
      </div>
      <ul className="mt-2 divide-y">
        {song && cur.inst && <StemRow cur={cur} kind="mix" buffer={cur.inst} color="stem-mix" title={t('stem.mix')} tag={t('res.recommended')} primary />}
        <StemRow cur={cur} kind="converted" buffer={cur.converted} color="stem-vocal" title={t('stem.converted')} primary={!song} />
        {song && cur.vocals && <StemRow cur={cur} kind="vocals" buffer={cur.vocals} color="stem-orig" title={t('stem.vocals')} />}
        {song && cur.inst && <StemRow cur={cur} kind="inst" buffer={cur.inst} color="stem-inst" title={t('stem.inst')} />}
      </ul>
      {song && <p className="text-caption text-muted-foreground mt-2">{t('res.mixNote')}</p>}
    </section>
  );
}

function Rerun({ result: r }: { result: Result }) {
  const t = useT();
  const { opts, prev, env } = useApp(useShallow((s) => ({ opts: s.opts, prev: s.prev, env: s.env })));
  const busy = useApp(selBusy);
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
          <p className="text-small font-semibold text-accent-foreground flex items-center gap-1.5"><Recycle className="size-4" />{sepAgain ? t('rerun.sepAgain') : r.mode === 'song' ? t('rerun.reuseSong') : t('rerun.reuseVoice')}</p>
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
      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <Button variant="ghost" size="sm" className="-ml-2" onClick={backToWork}><ArrowLeft />{t('res.back')}</Button>
        <span className="flex-1" />
        <p className="text-small text-muted-foreground flex items-center gap-1.5"><Info className="size-4" />{t('res.notSaved')}</p>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-5 items-start">
        <div className="space-y-5 min-w-0"><Hero result={result} cur={cur} /><Files cur={cur} /></div>
        <Rerun result={result} />
      </div>
    </main>
  );
}
