// (1) Song card and (2) voice model card (DESIGN 9.5, 9.6, 9.13).
import { useMemo, useState } from 'react';
import { ChevronDown, CircleCheck, CircleHelp, FolderHeart, Loader2, MicVocal, Music, Scissors, TriangleAlert, X } from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from './ui/command';
import { Switch } from './ui/switch';
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip';
import { Dropzone, pickFiles } from './Dropzone';
import { HelpTip, Segmented, StepCard } from './Field';
import { InlineError, NoticeButton } from './Notice';
import { AudioPlayer, usePlayer } from './Player';
import { AUDIO_ACCEPT, INDEX_ACCEPT, MODEL_ACCEPT, fmtBytes, fmtHz, fmtModelSize, fmtTime, fmtWhen, indexBytes } from '../lib/format';
import { cn } from '../lib/utils';
import { clearBacking, clearIndex, clearModel, clearSong, openMyModel, openSettings, saveMine, setOnnxSr, setOpts, tryClip } from '../controller';
import { selBusy, useApp, type Song } from '../store';
import { useT } from '../i18n';

const iconBtn = 'grid place-items-center size-8 rounded-lg hover:bg-card text-muted-foreground hover:text-foreground';

function SongPreview() {
  const t = useT();
  const song = useApp((s) => s.song)!;
  const range = useApp((s) => s.opts.range);
  const busy = useApp(selBusy);
  const stems = useMemo(() => (song.buffer ? { vocal: song.buffer } : undefined), [song.buffer]);
  const player = usePlayer(stems);
  if (!song.peaks || !song.duration) return null;
  return (
    <div className="mt-4">
      <div className="mb-2 flex items-center justify-end gap-2 min-h-8">
        {range ? (
          <span className="inline-flex items-center gap-1 h-8 pl-3 pr-1 rounded-full bg-accent text-accent-foreground text-caption font-semibold tnum">
            {t('opt.rangeChip', { a: fmtTime(range.start), b: fmtTime(range.end), n: Math.round(range.end - range.start) })}
            {!busy && <button type="button" className="grid place-items-center size-6 rounded-full hover:bg-card" aria-label={t('opt.rangeAll')} title={t('opt.rangeAll')} onClick={() => setOpts({ range: undefined })}><X className="size-3.5" /></button>}
          </span>
        ) : song.duration >= 30 && !busy && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button type="button" onClick={tryClip} className="inline-flex items-center gap-1.5 h-8 px-3 rounded-lg bg-secondary text-secondary-foreground text-small font-semibold hover:bg-muted">
                <Scissors className="size-4" />{t('opt.tryClip')}
              </button>
            </TooltipTrigger>
            <TooltipContent className="text-caption">{t('opt.tryClipHelp')}</TooltipContent>
          </Tooltip>
        )}
      </div>
      <AudioPlayer player={player} peaks={song.peaks} duration={song.duration} color="stem-orig" height={56} size="md" label={t('player.original')}
        region={range} onRegion={(r) => setOpts({ range: r })} />
    </div>
  );
}

/** One audio file: dropzone, error, or the file row. `slot` routes it to the backing slot in stems mode. */
function AudioSlot({ f, title, height, slot, onRemove, removeLabel }: {
  f?: Song; title: string; height: number; slot?: 'backing'; onRemove: () => void; removeLabel: string;
}) {
  const t = useT();
  const busy = useApp(selBusy);
  if (!f || f.status === 'parsing') {
    return <Dropzone icon={<Music className="size-6" />} title={title} hint={t('drop.songHint')} accept={AUDIO_ACCEPT} height={height} slot={slot}
      parsing={f?.status === 'parsing' ? t('drop.songParsing', { name: f.name }) : undefined} />;
  }
  if (f.status === 'error') {
    return <InlineError error={f.error} file={f.name} actions={<NoticeButton onClick={() => pickFiles(AUDIO_ACCEPT, false, slot)}>{t('common.pickOther')}</NoticeButton>} />;
  }
  return (
    <div className="flex items-center gap-3 rounded-xl bg-muted/60 p-3">
      <span className={cn('grid place-items-center size-10 rounded-lg bg-card shrink-0', slot ? 'text-stem-inst' : 'text-stem-mix')}><Music className="size-5" /></span>
      <div className="min-w-0 flex-1">
        <p className="font-semibold truncate" title={f.name}>{f.name}</p>
        <p className="text-caption text-muted-foreground tnum">
          {[fmtTime(f.duration ?? 0), f.sourceRate && fmtHz(f.sourceRate), f.channels === 1 ? t('meta.mono') : t('meta.stereo'), fmtBytes(f.size)].filter(Boolean).join(' · ')}
        </p>
      </div>
      {!busy && <>
        <button type="button" className="h-8 px-3 rounded-lg text-small font-semibold hover:bg-card" onClick={() => pickFiles(AUDIO_ACCEPT, false, slot)}>{t('common.replace')}</button>
        <button type="button" className={iconBtn} aria-label={removeLabel} onClick={onRemove}><X className="size-4" /></button>
      </>}
    </div>
  );
}

export function SongCard() {
  const t = useT();
  const song = useApp((s) => s.song);
  const backing = useApp((s) => s.backing);
  const mode = useApp((s) => s.opts.mode);
  const stems = mode === 'stems';
  const gap = stems && song?.duration && backing?.duration ? Math.abs(song.duration - backing.duration) : 0;
  return (
    <StepCard n={1} id="c1" title={t('card.song')}>
      {stems && <p className="text-small font-semibold mb-2">{t('stems.vocal')}</p>}
      <AudioSlot f={song} title={stems ? t('drop.vocalTitle') : t('drop.songTitle')} height={stems ? 112 : 236} onRemove={clearSong} removeLabel={t('song.remove')} />
      {song?.status === 'ready' && (song.duration ?? 0) > 600 && (
        <InlineError className="mt-3" tone="warning" title={t('err.AUDIO_LONG.title', { t: fmtTime(song.duration!) })} body={t('err.AUDIO_LONG.body')} />
      )}
      {song?.status === 'ready' && <SongPreview />}
      {stems && <>
        <p className="text-small font-semibold mt-4 mb-2">{t('stems.backing')}</p>
        <AudioSlot f={backing} slot="backing" title={t('drop.backingTitle')} height={112} onRemove={clearBacking} removeLabel={t('stems.removeBacking')} />
        {gap > 1 && <InlineError className="mt-3" tone="warning" title={t('stems.lenDiff', { n: Math.round(gap) })} />}
      </>}
      <div className="mt-5">
        <p className="text-small font-semibold mb-2 flex items-center gap-1">{t('mode.label')}<HelpTip text={t('mode.help')} /></p>
        <Segmented size="md" className="w-full" label={t('mode.label')} value={mode} onChange={(m) => setOpts({ mode: m })}
          items={[{ value: 'song', label: t('mode.song') }, { value: 'voice', label: t('mode.voice') }, { value: 'stems', label: t('mode.stems') }]} />
        <p className="text-caption text-muted-foreground mt-2">{t(mode === 'song' ? 'mode.songNote' : mode === 'voice' ? 'mode.voiceNote' : 'mode.stemsNote')}</p>
      </div>
    </StepCard>
  );
}

function MyModelPicker() {
  const t = useT();
  const list = useApp((s) => s.myModels) ?? [];
  const sizes = useApp((s) => s.cache?.entries);
  const [open, setOpen] = useState(false);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button type="button" className="inline-flex items-center gap-1 h-8 px-2.5 rounded-lg text-small font-semibold text-muted-foreground hover:bg-muted hover:text-foreground">
          <FolderHeart className="size-4" />{t('mine.button', { n: list.length })}
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 p-0">
        <Command>
          <CommandInput placeholder={t('mine.search')} />
          <CommandList>
            <CommandEmpty>{list.length ? t('mine.noMatch') : t('mine.empty')}</CommandEmpty>
            <CommandGroup>
              {list.map((m) => {
                const total = sizes?.find((e) => e.id === m.id)?.bytes ?? m.bytes;
                return (
                  <CommandItem key={m.id} value={`${m.name} ${m.memo ?? ''} ${m.id}`} onSelect={() => { setOpen(false); void openMyModel(m.id); }} className="flex items-center gap-2">
                    <MicVocal className="size-4 text-stem-vocal shrink-0" />
                    <span className="flex-1 min-w-0">
                      <span className="block truncate font-semibold">{m.name}</span>
                      {m.memo && <span className="block text-caption text-muted-foreground truncate">{m.memo}</span>}
                      <span className="block text-caption text-muted-foreground tnum truncate">{fmtModelSize(t, m.bytes, m.indexId ? total - m.bytes : undefined)} · {fmtWhen(t, m.lastUsed || m.savedAt)}</span>
                    </span>
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>
          <button type="button" className="w-full border-t px-3 h-10 text-left text-small font-semibold text-accent-foreground hover:bg-muted" onClick={() => { setOpen(false); openSettings('models'); }}>{t('mine.manage')}</button>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

function IndexRow() {
  const t = useT();
  const index = useApp((s) => s.index);
  const busy = useApp(selBusy);
  if (index?.status === 'error') {
    return (
      <InlineError className="mt-3" tone="warning" error={index.error} file={index.fileName}
        actions={<><NoticeButton onClick={() => pickFiles(INDEX_ACCEPT)}>{t('index.other')}</NoticeButton><NoticeButton onClick={clearIndex}>{t('index.without')}</NoticeButton></>} />
    );
  }
  return (
    <div className="mt-3 pt-3 border-t flex items-center gap-2 text-small">
      {!index ? <>
        <span className="size-4 rounded-full border-2 border-dashed border-muted-foreground/50 shrink-0" />
        <span className="flex-1 text-muted-foreground">{t('index.none')}</span>
        {!busy && <button type="button" className="text-small font-semibold text-accent-foreground" onClick={() => pickFiles(INDEX_ACCEPT)}>{t('index.add')}</button>}
      </> : index.status === 'parsing' ? <>
        <Loader2 className="size-4 animate-spin text-accent-foreground shrink-0" /><span className="flex-1 truncate">{t('index.parsing')}</span>
      </> : <>
        <CircleCheck className="size-4 text-success shrink-0" />
        <span className="truncate flex-1" title={index.fileName}>{t('index.linked')}</span>
        {!busy && <button type="button" className="text-small text-muted-foreground hover:text-foreground" onClick={clearIndex}>{t('common.remove')}</button>}
      </>}
    </div>
  );
}

/** "내 모델에 저장": on by default; the caption says what is (or will be) stored and how it went. */
function SaveRow({ size }: { size: string }) {
  const t = useT();
  const model = useApp((s) => s.model)!;
  const busy = useApp(selBusy);
  if (model.fromMine) return <p className="mt-3 pt-3 border-t text-caption text-muted-foreground flex items-center gap-1.5"><FolderHeart className="size-3.5" />{t('model.mineSaved')} · {size}</p>;
  const st = model.saveState;
  const caption = st === 'saving' ? <span className="inline-flex items-center gap-1"><Loader2 className="size-3 animate-spin" />{t('model.saving')}</span>
    : st === 'just' ? <span className="inline-flex items-center gap-1 text-success font-semibold"><CircleCheck className="size-3.5" />{t('model.savedJust')}</span>
    : st === 'quota' ? <span className="inline-flex items-center gap-1 text-warning font-semibold"><TriangleAlert className="size-3.5" />{t('model.saveQuota')} · <button type="button" className="underline underline-offset-2" onClick={() => openSettings('storage')}>{t('dl.manage')}</button></span>
    : st === 'failed' ? <span className="text-destructive">{t('toast.saveFailed')}</span>
    : model.saved ? t('model.savedNote', { size }) : size;
  return (
    <div className="mt-3 pt-3 border-t flex items-center gap-3">
      <label htmlFor="save-mine" className="flex-1 min-w-0 cursor-pointer">
        <span className="text-small font-semibold">{t('model.save')}</span>
        <span className="block text-caption text-muted-foreground tnum mt-0.5" aria-live="polite">{caption}</span>
      </label>
      <Switch id="save-mine" checked={!!model.saved} onCheckedChange={(v) => void saveMine(v)} disabled={busy || st === 'saving'} />
    </div>
  );
}

function ModelInfoCard() {
  const t = useT();
  const model = useApp((s) => s.model)!;
  const index = useApp((s) => s.index);
  const busy = useApp(selBusy);
  const [more, setMore] = useState(false); // DESIGN 9.5: not remembered
  const info = model.info!;
  const stored = useApp((s) => s.cache?.entries.find((e) => e.id === model.info?.id)?.bytes); // saved model + index, as settings shows it
  const ib = index?.status === 'ready' ? index.bytes ?? (stored && stored > info.bytes ? stored - info.bytes : indexBytes(index)) : undefined;
  const size = fmtModelSize(t, info.bytes, ib);
  const tag = 'px-2 py-1 rounded-md bg-muted';
  const source = model.fromMine ? t('model.fromMine') : model.fromZip ? t('model.fromZip', { name: model.fileName }) : undefined;
  const fileName = model.fromZip || model.fromMine ? info.name : model.fileName;
  const vectors = index?.info?.nVectors ?? info.index?.nVectors;
  return (
    <div className="rounded-xl border p-4">
      <div className="flex items-start gap-3">
        <span className="grid place-items-center size-10 rounded-lg bg-stem-vocal/15 text-stem-vocal shrink-0"><MicVocal className="size-5" /></span>
        <div className="min-w-0 flex-1">
          <p className="font-semibold truncate" title={info.name}>{info.name}</p>
          <div className="flex items-center gap-2">
            {info.f0
              ? <p className="text-caption font-semibold text-success flex items-center gap-1 flex-1 min-w-0"><CircleCheck className="size-3.5 shrink-0" /><span className="truncate">{t('model.ok')}</span></p>
              : <p className="text-caption font-semibold text-warning flex items-center gap-1 flex-1 min-w-0"><TriangleAlert className="size-3.5 shrink-0" /><span className="truncate">{t('model.nof0Note')}</span></p>}
            <button type="button" aria-expanded={more} onClick={() => setMore(!more)} className="inline-flex items-center gap-0.5 text-caption font-semibold text-accent-foreground shrink-0">
              {t('model.details')}<ChevronDown className={cn('size-3.5 transition-transform', more && 'rotate-180')} />
            </button>
          </div>
        </div>
        {!busy && <button type="button" className={cn(iconBtn, 'hover:bg-muted')} aria-label={t('model.remove')} onClick={clearModel}><X className="size-4" /></button>}
      </div>
      {more && (
        <div className="mt-3 rounded-lg bg-muted/60 p-3">
          <div className="flex flex-wrap gap-1.5 text-caption font-semibold">
            {info.source === 'onnx' && <span className={cn(tag, 'bg-card')}>ONNX</span>}
            <span className={cn(tag, 'bg-card')}>{info.version}</span>
            {!info.srUnknown && <span className={cn(tag, 'bg-card')}>{fmtHz(info.sr)}</span>}
            <span className={cn(tag, 'bg-card')}>{info.f0 ? t('model.f0') : t('model.nof0')}</span>
            {info.meta?.epoch !== undefined && <span className={cn(tag, 'bg-card')}>{t('model.epoch', { n: info.meta.epoch })}</span>}
            {info.meta?.vocoder && <span className={cn(tag, 'bg-card')}>{info.meta.vocoder}</span>}
          </div>
          <dl className="mt-2 space-y-0.5 text-caption text-muted-foreground tnum">
            {vectors !== undefined && <dd>{t('index.vectors', { n: vectors.toLocaleString() })}</dd>}
            <dd className="break-all">{ib ? t('model.file', { name: fileName, a: fmtBytes(info.bytes), b: fmtBytes(ib) }) : t('model.fileNoIndex', { name: fileName, a: fmtBytes(info.bytes) })}</dd>
            {source && <dd className="break-all">{source}</dd>}
          </dl>
        </div>
      )}
      {info.srUnknown && (
        <div className="mt-3 rounded-lg bg-warning-soft p-3">
          <p className="text-small font-semibold">{t('err.ONNX_NEED_SR.title')}</p>
          <p className="text-caption mt-0.5 mb-2">{t('err.ONNX_NEED_SR.body')}</p>
          <Segmented label={t('err.ONNX_NEED_SR.title')} value={undefined} onChange={(v) => void setOnnxSr(Number(v) as 32000 | 40000 | 48000)}
            items={[{ value: '32000', label: '32kHz' }, { value: '40000', label: '40kHz' }, { value: '48000', label: '48kHz' }]} />
        </div>
      )}
      <IndexRow />
      {index?.info?.downsampled && (
        <InlineError className="mt-3" tone="warning" title={t('err.INDEX_DOWNSAMPLED.title')} body={t('err.INDEX_DOWNSAMPLED.body', { n: index.info.downsampled.toLocaleString() })} />
      )}
      <SaveRow size={size} />
    </div>
  );
}

/** I-21: in-app explanation only, no external links or requests. */
function NoModelHelp() {
  const t = useT();
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button type="button" className="mt-2 inline-flex items-center gap-1 text-caption font-semibold text-accent-foreground hover:underline underline-offset-2">
          <CircleHelp className="size-3.5" />{t('model.noModel')}
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-80 text-small">{t('model.noModelBody')}</PopoverContent>
    </Popover>
  );
}

export function ModelSection() {
  const t = useT();
  const model = useApp((s) => s.model);
  const index = useApp((s) => s.index);
  return (
    <StepCard n={2} id="c2" title={t('card.model')} aside={<MyModelPicker />}>
      {!model || model.status === 'parsing' ? <>
        <Dropzone icon={<MicVocal className="size-6" />} title={t('drop.modelTitle')} hint={t('drop.modelHint')} accept={MODEL_ACCEPT} multiple height={176}
          parsing={model?.status === 'parsing' ? t('drop.modelParsing', { name: model.fileName }) : undefined} />
        {index && !model ? <IndexRow /> : <p className="text-caption text-muted-foreground mt-3">{t('drop.indexNote')}</p>}
        {!model && <NoModelHelp />}
      </> : model.status === 'error' ? (
        <InlineError error={model.error} file={model.fileName} actions={<NoticeButton onClick={() => pickFiles(MODEL_ACCEPT, true)}>{t('common.pickOther')}</NoticeButton>} />
      ) : <ModelInfoCard />}
    </StepCard>
  );
}
