// (1) Song card and (2) voice model card.
import { useEffect, useMemo, useState } from 'react';
import { CircleCheck, FolderHeart, Loader2, MicVocal, Music, X } from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from './ui/command';
import { Switch } from './ui/switch';
import { Dropzone, pickFiles } from './Dropzone';
import { HelpTip, Segmented, StepCard } from './Field';
import { InlineError, NoticeButton } from './Notice';
import { AudioPlayer, usePlayer } from './Player';
import type { SavedModel } from '@/shared/types';
import { AUDIO_ACCEPT, INDEX_ACCEPT, MODEL_ACCEPT, fmtBytes, fmtHz, fmtTime } from '../lib/format';
import { cn } from '../lib/utils';
import { clearIndex, clearModel, clearSong, engine, openMyModel, saveMine, setOnnxSr, setOpts } from '../controller';
import { selBusy, set, useApp } from '../store';
import { useT } from '../i18n';


const iconBtn = 'grid place-items-center size-8 rounded-lg hover:bg-card text-muted-foreground hover:text-foreground';

function SongPreview() {
  const t = useT();
  const song = useApp((s) => s.song)!;
  const range = useApp((s) => s.opts.range);
  const stems = useMemo(() => (song.buffer ? { vocal: song.buffer } : undefined), [song.buffer]);
  const player = usePlayer(stems);
  if (!song.peaks || !song.duration) return null;
  return (
    <div className="mt-4">
      {range && (
        <div className="mb-2 flex items-center gap-2">
          <span className="inline-flex items-center h-7 px-2.5 rounded-full bg-accent text-accent-foreground text-caption font-semibold tnum">
            {t('opt.rangeChip', { a: fmtTime(range.start), b: fmtTime(range.end), n: Math.round(range.end - range.start) })}
          </span>
          <button type="button" className="text-caption font-semibold text-muted-foreground hover:text-foreground" onClick={() => setOpts({ range: undefined })}>{t('opt.rangeAll')}</button>
        </div>
      )}
      <AudioPlayer player={player} peaks={song.peaks} duration={song.duration} color="stem-orig" height={56} size="md" label={t('player.original')}
        region={range} onRegion={(r) => setOpts({ range: r })} />
      <p className="text-caption text-muted-foreground mt-1 pl-12">{t('opt.dragHint')}</p>
    </div>
  );
}

export function SongCard() {
  const t = useT();
  const song = useApp((s) => s.song);
  const mode = useApp((s) => s.opts.mode);
  const busy = useApp(selBusy);
  return (
    <StepCard n={1} id="c1" title={t('card.song')}>
      {!song || song.status === 'parsing' ? (
        <Dropzone icon={<Music className="size-6" />} title={t('drop.songTitle')} hint={t('drop.songHint')} accept={AUDIO_ACCEPT} height={236}
          parsing={song?.status === 'parsing' ? t('drop.songParsing', { name: song.name }) : undefined} />
      ) : song.status === 'error' ? (
        <InlineError error={song.error} file={song.name} actions={<NoticeButton onClick={() => pickFiles(AUDIO_ACCEPT)}>{t('common.pickOther')}</NoticeButton>} />
      ) : (
        <div>
          <div className="flex items-center gap-3 rounded-xl bg-muted/60 p-3">
            <span className="grid place-items-center size-10 rounded-lg bg-card text-stem-mix shrink-0"><Music className="size-5" /></span>
            <div className="min-w-0 flex-1">
              <p className="font-semibold truncate" title={song.name}>{song.name}</p>
              <p className="text-caption text-muted-foreground tnum">
                {[fmtTime(song.duration ?? 0), song.sourceRate && fmtHz(song.sourceRate), song.channels === 1 ? t('meta.mono') : t('meta.stereo'), fmtBytes(song.size)].filter(Boolean).join(' · ')}
              </p>
            </div>
            {!busy && <>
              <button type="button" className="h-8 px-3 rounded-lg text-small font-semibold hover:bg-card" onClick={() => pickFiles(AUDIO_ACCEPT)}>{t('common.replace')}</button>
              <button type="button" className={iconBtn} aria-label={t('song.remove')} onClick={clearSong}><X className="size-4" /></button>
            </>}
          </div>
          {(song.duration ?? 0) > 600 && (
            <InlineError className="mt-3" tone="warning" title={t('err.AUDIO_LONG.title', { t: fmtTime(song.duration!) })} body={t('err.AUDIO_LONG.body')} />
          )}
          <SongPreview />
        </div>
      )}
      <div className="mt-5">
        <p className="text-small font-semibold mb-2 flex items-center gap-1">{t('mode.label')}<HelpTip text={t('mode.help')} /></p>
        <Segmented size="md" className="w-full" label={t('mode.label')} value={mode} onChange={(m) => setOpts({ mode: m })}
          items={[{ value: 'song', label: <>{t('mode.song')} <span className="font-normal text-muted-foreground">{t('mode.songSub')}</span></> }, { value: 'voice', label: t('mode.voice') }]} />
        <p className="text-caption text-muted-foreground mt-2">{mode === 'song' ? t('mode.songNote') : t('mode.voiceNote')}</p>
      </div>
    </StepCard>
  );
}

function MyModelPicker() {
  const t = useT();
  const version = useApp((s) => s.myModelsVersion);
  const [list, setList] = useState<SavedModel[]>([]);
  const [open, setOpen] = useState(false);
  const ready = useApp((s) => s.engineReady);
  useEffect(() => { if (ready) void engine.myModels.list().then(setList).catch(() => setList([])); }, [version, open, ready]);
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
              {list.map((m) => (
                <CommandItem key={m.id} value={`${m.name} ${m.id}`} onSelect={() => { setOpen(false); void openMyModel(m.id); }} className="flex items-center gap-2">
                  <MicVocal className="size-4 text-stem-vocal" />
                  <span className="flex-1 truncate">{m.name}</span>
                  <span className="text-caption text-muted-foreground">{m.version} · {m.sr / 1000}k{m.indexId ? ' · index' : ''}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
          <button type="button" className="w-full border-t px-3 h-10 text-left text-small font-semibold text-primary hover:bg-muted" onClick={() => { setOpen(false); set({ settingsTab: 'models' }); }}>{t('mine.manage')}</button>
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
        {!busy && <button type="button" className="text-small font-semibold text-primary" onClick={() => pickFiles(INDEX_ACCEPT)}>{t('index.add')}</button>}
      </> : index.status === 'parsing' ? <>
        <Loader2 className="size-4 animate-spin text-primary shrink-0" /><span className="flex-1 truncate">{t('index.parsing')}</span>
      </> : <>
        <CircleCheck className="size-4 text-success shrink-0" />
        <span className="truncate flex-1" title={index.fileName}>
          {index.info ? t('index.linked', { n: index.info.nVectors.toLocaleString() }) : t('index.linkedPacked')}
        </span>
        {!busy && <button type="button" className="text-small text-muted-foreground hover:text-foreground" onClick={clearIndex}>{t('common.remove')}</button>}
      </>}
    </div>
  );
}

function ModelInfoCard() {
  const t = useT();
  const model = useApp((s) => s.model)!;
  const index = useApp((s) => s.index);
  const busy = useApp(selBusy);
  const info = model.info!;
  const tag = 'px-2 py-1 rounded-md bg-muted';
  const source = model.fromMine ? t('model.fromMine') : model.fromZip ? t('model.fromZip', { name: model.fileName }) : model.fileName;
  return (
    <div className="rounded-xl border p-4">
      <div className="flex items-start gap-3">
        <span className="grid place-items-center size-10 rounded-lg bg-stem-vocal/15 text-stem-vocal shrink-0"><MicVocal className="size-5" /></span>
        <div className="min-w-0 flex-1">
          <p className="font-semibold truncate" title={info.name}>{info.name}</p>
          <p className="text-caption text-muted-foreground tnum truncate" title={source}>{source} · {fmtBytes(info.bytes)}</p>
        </div>
        {!busy && <button type="button" className={cn(iconBtn, 'hover:bg-muted')} aria-label={t('model.remove')} onClick={clearModel}><X className="size-4" /></button>}
      </div>
      <div className="flex flex-wrap gap-1.5 mt-3 text-caption font-semibold">
        {info.source === 'onnx' && <span className={cn(tag, 'bg-accent text-accent-foreground')}>ONNX</span>}
        <span className={tag}>{info.version}</span>
        {!info.srUnknown && <span className={tag}>{fmtHz(info.sr)}</span>}
        <span className={cn(tag, !info.f0 && 'bg-warning-soft text-warning')}>{info.f0 ? t('model.f0') : t('model.nof0')}</span>
        <span className={tag}>{t('model.spk', { n: info.nSpk })}</span>
        {info.meta?.vocoder && <span className={tag}>{info.meta.vocoder}</span>}
        {info.meta?.epoch !== undefined && <span className={tag}>{t('model.epoch', { n: info.meta.epoch })}</span>}
      </div>
      {!info.f0 && <p className="text-caption text-warning mt-2">{t('model.nof0Note')}</p>}
      {info.srUnknown && (
        <div className="mt-3 rounded-lg bg-warning-soft p-3">
          <p className="text-small font-semibold">{t('err.ONNX_NEED_SR.title')}</p>
          <p className="text-caption text-muted-foreground mt-0.5 mb-2">{t('err.ONNX_NEED_SR.body')}</p>
          <Segmented label={t('err.ONNX_NEED_SR.title')} value={'' as string} onChange={(v) => void setOnnxSr(Number(v) as 32000 | 40000 | 48000)}
            items={[{ value: '32000', label: '32kHz' }, { value: '40000', label: '40kHz' }, { value: '48000', label: '48kHz' }]} />
        </div>
      )}
      <IndexRow />
      {index?.info?.downsampled && (
        <InlineError className="mt-3" tone="warning" title={t('err.INDEX_DOWNSAMPLED.title')} body={t('err.INDEX_DOWNSAMPLED.body', { n: index.info.downsampled.toLocaleString() })} />
      )}
      <div className="mt-3 pt-3 border-t flex items-center gap-3">
        <label htmlFor="save-mine" className="flex-1 min-w-0 cursor-pointer">
          <span className="text-small font-semibold">{t('model.save')}</span>
          <span className="block text-caption text-muted-foreground">{t('model.saveNote')}</span>
        </label>
        <Switch id="save-mine" checked={!!model.saved} onCheckedChange={(v) => void saveMine(v)} disabled={busy} />
      </div>
    </div>
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
      </> : model.status === 'error' ? (
        <InlineError error={model.error} file={model.fileName} actions={<NoticeButton onClick={() => pickFiles(MODEL_ACCEPT, true)}>{t('common.pickOther')}</NoticeButton>} />
      ) : <ModelInfoCard />}
    </StepCard>
  );
}
