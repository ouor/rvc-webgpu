// ActionBar (S1 bottom), S2 public-model download dialog, S3 progress panel.
import { useMemo, type ReactNode } from 'react';
import { useNow } from '../lib/hooks';
import { Activity, Boxes, Brain, Circle, CircleCheck, CircleDashed, CircleMinus, CircleX, Download as DownloadIcon, Info, Loader2, Lock, Recycle, Split, TriangleAlert } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog';
import { Button } from './ui/button';
import { Bar } from './Bar';
import { InlineError } from './Notice';
import { PlayButton, usePlayer } from './Player';
import type { DownloadItem } from '@/shared/types';
import { estimateRun, estimateStage, fmtBytes, fmtEta, fmtLeft, fmtTime, formatChanges, changedKeys, hostOf, type RunStage } from '../lib/format';
import { roleOf, type Role } from '../lib/catalog';
import { useShallow } from 'zustand/react/shallow';
import { cn } from '../lib/utils';
import { cancelJob, closeDownload, copyEnvInfo, exportSeparated, openSettings, recover, showDownload, start, startDownload, stopDownload } from '../controller';
import { STEPS, selMissingBytes, selModelBaseUrl, selSeconds, useApp, type Step, type StepInfo } from '../store';
import { useT } from '../i18n';

// ---------------- ActionBar ----------------

export function ActionBar() {
  const t = useT();
  const { song, model, index, opts, env, dl, engineReady, envError } = useApp(useShallow((s) => ({ song: s.song, model: s.model, index: s.index, opts: s.opts, env: s.env, dl: s.dl, engineReady: s.engineReady, envError: s.envError })));
  const seconds = useApp(selSeconds);
  const need = useApp(selMissingBytes);
  const voice = opts.mode === 'voice';
  const cta = voice ? t('cta.voice') : t('cta.song');
  let reason: ReactNode | undefined, sub: ReactNode | undefined, title: ReactNode;
  const fileError = song?.status === 'error' || model?.status === 'error';
  if (fileError) { title = t('bar.fixFiles'); sub = t('bar.fixFilesSub'); reason = title; }
  else if (song?.status !== 'ready' || model?.status !== 'ready') {
    title = song?.status === 'parsing' || model?.status === 'parsing' || index?.status === 'parsing' ? t('bar.parsing') : t('bar.empty');
    reason = title;
    sub = (
      <span className="flex gap-3">
        {[[song?.status === 'ready', t('card.song')], [model?.status === 'ready', t('card.model')]].map(([ok, label]) => (
          <span key={String(label)} className={cn('inline-flex items-center gap-1', ok && 'text-success')}>
            {ok ? <CircleCheck className="size-3.5" /> : <Circle className="size-3.5" />}{label}
          </span>
        ))}
      </span>
    );
  } else if (model.info?.srUnknown) { title = t('bar.needSr'); reason = title; }
  else if (envError) { title = t('banner.initTitle'); reason = title; }
  else if (!engineReady) { title = t('bar.preparing'); reason = title; }
  else {
    const stages: RunStage[] = voice ? ['f0', 'features', 'convert'] : ['separate', 'f0', 'features', 'convert'];
    const eta = fmtEta(t, estimateRun(seconds, env?.backend ?? 'webgpu', stages));
    title = <>{voice ? t('bar.voiceFlow') : t('bar.songFlow')} <span className="text-small font-normal text-muted-foreground tnum">· {t('bar.eta', { eta })}{opts.range ? ` · ${t('bar.rangeBased')}` : ''}</span></>;
    sub = need > 0
      ? <span className="inline-flex items-center gap-1.5"><DownloadIcon className="size-3.5" />{voice ? t('bar.skipSep') + ' ' : ''}{t('bar.firstDl', { size: fmtBytes(need) })}</span>
      : voice ? t('bar.skipSep') : t('bar.cached');
  }
  const loadingDl = dl?.status === 'loading' && !dl.open;
  return (
    <div className="fixed bottom-5 inset-x-0 z-30 px-6 pointer-events-none">
      <div className="mx-auto max-w-[1232px] rounded-2xl bg-card shadow-float border px-5 py-4 flex items-center gap-4 pointer-events-auto">
        <div className="flex-1 min-w-0" id="bar-why">
          {loadingDl ? <>
            <p className="font-semibold">{t('dl.barLoading', { p: Math.floor((dl.done / Math.max(1, dl.total)) * 100) })}</p>
            <Bar className="mt-2 max-w-md" value={(dl.done / Math.max(1, dl.total)) * 100} label={t('dl.loadingTitle')} />
          </> : <>
            <p className="font-semibold flex items-center gap-2 flex-wrap">{title}</p>
            {sub && <div className="text-small text-muted-foreground mt-1">{sub}</div>}
          </>}
        </div>
        {loadingDl
          ? <Button variant="secondary" onClick={() => showDownload(true)}>{t('dl.show')}</Button>
          : <Button size="cta" disabled={!!reason || !!dl}
            aria-describedby={reason ? 'bar-why' : undefined} onClick={() => void start()}>{cta}</Button>}
      </div>
    </div>
  );
}

// ---------------- S2 download dialog ----------------

const ROLE_ICON: Record<Role, typeof Split> = { 'pm.separate': Split, 'pm.f0': Activity, 'pm.features': Brain, 'pm.template': Boxes };

function DlItem({ item, got, state, failed }: { item: DownloadItem; got: number; state: 'wait' | 'run' | 'done' | 'confirm'; failed?: boolean }) {
  const t = useT();
  const role = roleOf(item.id);
  const Icon = ROLE_ICON[role];
  const [name, sub] = [t(role), item.label];
  const pct = Math.min(100, (got / item.bytes) * 100);
  return (
    <li className="flex items-center gap-3 py-3">
      <span className="grid place-items-center size-9 rounded-lg bg-muted text-muted-foreground shrink-0"><Icon className="size-4" /></span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2 text-small">
          <span className="font-semibold truncate">{name} <span className="font-normal text-muted-foreground">{sub}</span></span>
          <span className="text-muted-foreground tnum shrink-0">{state === 'run' ? `${fmtBytes(got)} / ` : ''}{item.bytes < 1e6 ? t('dl.underMb') : fmtBytes(item.bytes)}</span>
        </div>
        {state === 'run' && <Bar className="mt-1.5" value={pct >= 100 ? undefined : pct} label={name} />}
        {state === 'run' && pct >= 100 && <p className="text-caption text-muted-foreground mt-1">{t('dl.verify')}</p>}
        {state === 'wait' && !failed && <p className="text-caption text-muted-foreground mt-0.5">{t('dl.waiting')}</p>}
        {failed && state !== 'done' && <p className="text-caption text-destructive mt-0.5">{t('dl.itemFailed')}</p>}
      </div>
      {state === 'run' && pct < 100 && <span className="text-small font-semibold tnum w-10 text-right">{Math.floor(pct)}%</span>}
      {state === 'done' && <CircleCheck className="size-5 text-success shrink-0" />}
    </li>
  );
}

export function DownloadDialog() {
  const t = useT();
  const dl = useApp((s) => s.dl);
  const env = useApp((s) => s.env);
  const now = useNow(dl?.status === 'loading');
  const baseUrl = useApp(selModelBaseUrl);
  if (!dl) return null;
  const close = (o: boolean) => { if (o) return; if (dl.status === 'loading') showDownload(false); else closeDownload(); };
  const rows = dl.items.map((item) => {
    const got = dl.got?.[item.id] ?? 0;
    const state: 'wait' | 'run' | 'done' | 'confirm' = ['confirm', 'storage', 'offline'].includes(dl.status) ? 'confirm'
      : item.id === dl.cur ? (dl.status === 'loading' ? 'run' : 'wait') : dl.got?.[item.id] !== undefined ? 'done' : 'wait';
    return { item, got, state };
  });
  const firstFailed = dl.status === 'failed' ? rows.find((r) => r.state !== 'done') : undefined;
  const rate = dl.startedAt && dl.done > 0 ? dl.done / ((now - dl.startedAt) / 1000) : 0;
  const left = rate > 0 ? (dl.total - dl.done) / rate : undefined;
  const free = env ? Math.max(0, env.storage.quota - env.storage.usage) : 0;
  const head = {
    confirm: [t('dl.title'), t('dl.body')],
    loading: [t('dl.loadingTitle'), t('dl.loadingBody')],
    failed: [t('dl.failedTitle'), dl.error?.code === 'CACHE_INTEGRITY' ? t('err.CACHE_INTEGRITY.body') : t('dl.failedBody')],
    storage: [t('dl.storageTitle'), t('dl.storageBody', { need: fmtBytes(dl.total), free: fmtBytes(free) })],
    offline: [t('dl.offlineTitle'), t('dl.offlineBody')],
  }[dl.status];
  return (
    <Dialog open={dl.open} onOpenChange={close}>
      <DialogContent className="sm:max-w-[520px] rounded-2xl p-6 gap-0" aria-describedby="dl-desc">
        <DialogTitle className="text-h1">{head[0]}</DialogTitle>
        <DialogDescription id="dl-desc" className="text-body text-muted-foreground mt-2">{head[1]}</DialogDescription>
        <ul className="mt-4 divide-y border-y">
          {rows.map((r) => <DlItem key={r.item.id} item={r.item} got={r.got} state={r.state} failed={firstFailed === r} />)}
        </ul>
        {dl.status === 'loading' ? (
          <div className="mt-4">
            <div className="flex justify-between text-small tnum mb-2">
              <span className="font-semibold">{fmtBytes(dl.done)} / {fmtBytes(dl.total)}</span>
              <span className="text-muted-foreground">{left !== undefined ? t('dl.left', { t: fmtLeft(t, left) }) : t('eta.calc')}</span>
            </div>
            <Bar className="h-2" value={(dl.done / Math.max(1, dl.total)) * 100} label={t('dl.loadingTitle')} />
          </div>
        ) : (
          <div className="mt-4 flex justify-between text-body"><span className="text-muted-foreground">{t('dl.total')}</span><span className="font-bold tnum">{t('dl.about', { size: fmtBytes(dl.total) })}</span></div>
        )}
        {dl.status === 'confirm' && <p className="mt-4 flex items-center gap-1.5 text-caption text-muted-foreground"><Lock className="size-3.5" />{t('dl.source', { host: hostOf(baseUrl) })}</p>}
        {dl.error && dl.status === 'failed' && <InlineError className="mt-4" error={dl.error} />}
        <div className="mt-6 flex justify-end gap-2">
          {dl.status === 'confirm' && <>
            <Button variant="ghost" onClick={closeDownload}>{t('dl.later')}</Button>
            <Button onClick={() => void startDownload()}>{t('dl.start')}</Button>
          </>}
          {dl.status === 'loading' && <Button variant="secondary" onClick={() => void stopDownload()}>{t('dl.stop')}</Button>}
          {dl.status === 'failed' && <>
            <Button variant="ghost" onClick={closeDownload}>{t('common.close')}</Button>
            <Button onClick={() => void startDownload(true)}>{t('dl.retry')}</Button>
          </>}
          {dl.status === 'storage' && <>
            <Button variant="ghost" onClick={closeDownload}>{t('common.close')}</Button>
            <Button onClick={() => { closeDownload(); openSettings('storage'); }}>{t('dl.manage')}</Button>
          </>}
          {dl.status === 'offline' && <>
            <Button variant="ghost" onClick={closeDownload}>{t('common.close')}</Button>
            <Button onClick={() => void startDownload(true)}>{t('common.retry')}</Button>
          </>}
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ---------------- S3 progress panel ----------------

function StemButtons() {
  const t = useT();
  const stems = useApp((s) => s.stems);
  const v = useMemo(() => (stems ? { vocal: stems.vocals } : undefined), [stems]);
  const i = useMemo(() => (stems ? { vocal: stems.inst } : undefined), [stems]);
  const pv = usePlayer(v), pi = usePlayer(i);
  if (!stems) return null;
  return (
    <div className="flex items-center gap-1.5 mt-2">
      <span className="inline-flex items-center gap-1 text-caption font-semibold"><PlayButton player={pv} size="sm" label={t('stem.vocals')} className="size-7" />{t('stem.vocalsShort')}</span>
      <span className="inline-flex items-center gap-1 text-caption font-semibold"><PlayButton player={pi} size="sm" label={t('stem.inst')} className="size-7" />{t('stem.instShort')}</span>
      <button type="button" className="grid place-items-center size-7 rounded-lg hover:bg-card text-muted-foreground hover:text-foreground" aria-label={t('stem.saveSeparated')} title={t('stem.saveSeparated')} onClick={() => void exportSeparated()}>
        <DownloadIcon className="size-4" />
      </button>
    </div>
  );
}

function StepItem({ k, info, cancelling, rerunKeyOnly }: { k: Step; info: StepInfo; cancelling: boolean; rerunKeyOnly: boolean }) {
  const t = useT();
  const name = t(`step.${k}`);
  const pct = info.total ? Math.floor((info.done / info.total) * 100) : 0;
  const active = info.state === 'run' || info.state === 'load';
  const Icon = { wait: CircleDashed, load: Loader2, run: Loader2, done: CircleCheck, reused: Recycle, skipped: CircleMinus, error: CircleX, cancelled: CircleMinus }[info.state];
  let line: ReactNode;
  if (active && cancelling) line = t('step.stopping');
  else if (info.state === 'load') line = t('step.loading', { m: t(`step.model.${k}`) });
  else if (info.state === 'run') line = k === 'separate' ? t('step.patch', { n: info.done, m: info.total, p: pct }) : t('step.chunk', { n: info.done, m: info.total, p: pct });
  else if (info.state === 'done') line = info.ms ? t('step.doneIn', { t: fmtTime(info.ms / 1000) }) : t('step.done');
  else if (info.state === 'reused') line = k === 'f0' && rerunKeyOnly ? t('step.reusedKey') : t('step.reused');
  else if (info.state === 'skipped') line = t('step.skipped');
  else if (info.state === 'error') line = info.total ? t('step.stoppedAt', { n: info.done + 1, m: info.total }) : t('step.stopped');
  else line = t('step.wait');
  return (
    <li className={cn('rounded-xl p-3 min-w-0', active && 'bg-accent', info.state === 'error' && 'bg-destructive-soft', (info.state === 'wait' || info.state === 'skipped') && 'text-muted-foreground')} aria-current={active ? 'step' : undefined}>
      <div className="flex items-center gap-1.5">
        <Icon className={cn('size-4 shrink-0', active && 'animate-spin text-primary', info.state === 'done' && 'text-success', info.state === 'reused' && 'text-primary', info.state === 'error' && 'text-destructive')} />
        <span className={cn('text-small font-semibold truncate', info.state === 'skipped' && 'line-through')}>{name}</span>
      </div>
      <p className="text-caption mt-1 tnum truncate">{line}</p>
      {info.state === 'run' && !cancelling && (
        <Bar className="mt-2 bg-card" value={pct} label={name} />
      )}
      {info.state === 'load' && !cancelling && <Bar className="mt-2 bg-card" label={name} />}
      {k === 'separate' && (info.state === 'done' || info.state === 'reused') && <StemButtons />}
    </li>
  );
}

export function ProgressPanel() {
  const t = useT();
  const job = useApp((s) => s.job);
  const env = useApp((s) => s.env);
  const result = useApp((s) => s.result);
  const now = useNow(!!job && job.status !== 'error');
  if (!job) return null;
  const steps = STEPS.filter((k) => job.steps[k].state !== 'skipped');
  const curIdx = job.current ? steps.indexOf(job.current) : 0;
  const cur = job.current ? job.steps[job.current] : undefined;
  const backend = job.fallback ? 'wasm' : env?.backend ?? 'webgpu';
  let left: number | undefined;
  if (cur && job.current !== 'mix') {
    const curLeft = cur.state === 'run' && cur.done > 0 && cur.startedAt ? ((now - cur.startedAt) / 1000 / cur.done) * (cur.total - cur.done) : undefined;
    const pending = STEPS.filter((k) => k !== 'mix' && job.steps[k].state === 'wait').reduce((a, k) => a + estimateStage(k as RunStage, job.seconds, backend), 0);
    left = curLeft === undefined ? undefined : curLeft + pending;
  }
  const elapsed = (now - job.startedAt) / 1000;
  const err = job.status === 'error' ? job.error : undefined;
  const sepOk = job.steps.separate.state === 'done' || job.steps.separate.state === 'reused';
  const changed = job.change ? changedKeys(job.change.from, job.change.to) : [];
  const keyOnly = !!result && changed.length === 1 && changed[0] === 'key';
  const changes = job.change && formatChanges(t, job.change.from, job.change.to);
  const title = err ? (err.code === 'OOM' ? t('err.OOM.title') : t('prog.failed'))
    : job.status === 'cancelling' ? t('prog.cancelling')
    : job.rerun ? t('prog.rerun') : job.mode === 'voice' ? t('prog.voice') : t('prog.song');
  const actions = err && (
    <>
      {err.code === 'OOM' && <Button size="sm" onClick={() => void recover('low')}>{t('prog.lowMem')}</Button>}
      {err.code === 'GPU_LOST' && <Button size="sm" onClick={() => void recover('wasm')}>{t('prog.useWasm')}</Button>}
      {err.code === 'INDEX_DIM_MISMATCH' && <Button size="sm" onClick={() => void recover('noIndex')}>{t('index.without')}</Button>}
      {err.code !== 'OOM' && <Button variant={err.code === 'GPU_LOST' || err.code === 'INDEX_DIM_MISMATCH' ? 'secondary' : 'default'} size="sm" onClick={() => void recover('retry')}>{t('common.retry')}</Button>}
      {sepOk && <Button variant="secondary" size="sm" onClick={() => void exportSeparated()}><DownloadIcon className="size-4" />{t('prog.saveSeparated')}</Button>}
      {err.code === 'INTERNAL' && <Button variant="secondary" size="sm" onClick={copyEnvInfo}>{t('about.copyEnv')}</Button>}
      <Button variant="ghost" size="sm" onClick={() => void cancelJob()}>{t('common.close')}</Button>
    </>
  );
  return (
    <section className="fixed bottom-5 inset-x-0 z-30 px-6" aria-labelledby="pp-t">
      <div className="mx-auto max-w-[1232px] rounded-2xl bg-card shadow-float border p-5 animate-in slide-in-from-bottom-4 fade-in duration-200">
        <div className="flex items-start gap-4">
          <div className="flex-1 min-w-0">
            <h2 id="pp-t" className="text-h2">{title}{changes && !err && <span className="font-normal text-muted-foreground"> · {changes}</span>}</h2>
            {!err && (
              <p className="text-small text-muted-foreground mt-0.5 tnum">
                {job.status === 'cancelling' ? t('prog.cancellingSub') : <>
                  {curIdx + 1}/{steps.length} {t(`step.${job.current ?? steps[0]}`)} · {t('prog.elapsed', { t: fmtTime(elapsed) })} · {left === undefined ? t('prog.leftCalc') : t('prog.left', { t: fmtLeft(t, left) })}
                </>}
              </p>
            )}
            <span className="sr-only" aria-live="polite">{job.current && t(`step.${job.current}`)}</span>
          </div>
          {!err && (job.status === 'cancelling'
            ? <Button variant="secondary" disabled><Loader2 className="animate-spin" />{t('prog.cancellingBtn')}</Button>
            : <Button variant="secondary" onClick={() => void cancelJob()}>{t('common.cancel')}</Button>)}
        </div>
        {job.fallback && !err && (
          <div className="mt-3 flex items-start gap-2 rounded-lg bg-warning-soft px-3 py-2 text-small">
            <TriangleAlert className="size-4 text-warning mt-0.5 shrink-0" />
            <p><b>{t('prog.fallback')}</b> <span className="text-muted-foreground">{t('prog.fallbackSub')}</span></p>
          </div>
        )}
        <ol className="mt-4 grid gap-2" style={{ gridTemplateColumns: 'repeat(5, minmax(0, 1fr))' }}>
          {STEPS.map((k) => <StepItem key={k} k={k} info={job.steps[k]} cancelling={job.status === 'cancelling'} rerunKeyOnly={keyOnly} />)}
        </ol>
        {err && <InlineError className="mt-4" error={err} actions={actions} />}
        {!err && <p className="mt-3 flex items-center gap-1.5 text-caption text-muted-foreground"><Info className="size-3.5" />{t('prog.note')}</p>}
      </div>
    </section>
  );
}

