// S5 settings dialog (vertical tabs).
import { useEffect, useRef, useState } from 'react';
import { Check, Clipboard, FolderHeart, FolderInput, HardDrive, Info, Loader2, Lock, MicVocal, Pencil, ShieldCheck, SlidersHorizontal, Trash2, X } from 'lucide-react';
import { Dialog as D, Tabs } from 'radix-ui';
import { Button } from './ui/button';
import { Segmented, SwitchRow } from './Field';
import type { BackendPref, CacheEntry } from '@/shared/types';
import { fmtBytes, fmtHz, hostOf } from '../lib/format';
import { roleOf } from '../lib/catalog';
import { cn } from '../lib/utils';
import { clearCache, copyEnvInfo, importCache, openMyModel, openSettings, removeCacheItem, removeMine, renameMine, setBackend, setModelBaseUrl, showNoticeAgain, updateSettings, verifyCacheItem } from '../controller';
import { DEFAULT_BASE_URL, selBusy, selModelBaseUrl, useApp, type SettingsTab } from '../store';
import { useT } from '../i18n';
import { version } from '../../../package.json';

const h3 = 'text-h3';
const fmtDate = (ms: number, lang: string) => new Date(ms).toLocaleDateString(lang, { year: 'numeric', month: 'short', day: 'numeric' });

function StatusChip({ status }: { status: CacheEntry['status'] | 'checking' }) {
  const t = useT();
  const tone = { verified: 'bg-success/15 text-success', incomplete: 'bg-warning-soft text-warning', corrupt: 'bg-destructive-soft text-destructive', checking: 'bg-muted text-muted-foreground' }[status];
  return (
    <span className={cn('inline-flex items-center gap-1 h-6 px-2 rounded-full text-caption font-semibold shrink-0', tone)}>
      {status === 'checking' ? <Loader2 className="size-3 animate-spin" /> : status === 'verified' ? <Check className="size-3" /> : null}{t(`cache.${status}`)}
    </span>
  );
}

function RenameInput({ name, onDone }: { name: string; onDone: (name?: string) => void }) {
  const t = useT();
  const [v, setV] = useState(name);
  const ended = useRef(false); // Escape then the unmount blur must not commit
  const end = (n?: string) => { if (!ended.current) { ended.current = true; onDone(n); } };
  return (
    <input autoFocus value={v} maxLength={80} aria-label={t('mine.rename')} onChange={(e) => setV(e.target.value)} onFocus={(e) => e.target.select()}
      onBlur={() => end(v.trim())} onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') { e.stopPropagation(); end(); } }}
      className="w-full h-7 rounded-md border border-input bg-card px-2 text-small font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" />
  );
}

function General() {
  const t = useT();
  const settings = useApp((s) => s.settings);
  const env = useApp((s) => s.env);
  const busy = useApp(selBusy);
  const setS = updateSettings;
  const pick = (b: BackendPref) => setBackend(b);
  const cards: [BackendPref, string, string][] = [
    ['auto', t('set.auto'), env ? t('set.autoNow', { b: env.backend === 'webgpu' ? 'WebGPU' : 'WASM' }) : ''],
    ['webgpu', 'WebGPU', t('set.webgpuSub')],
    ['wasm', 'WASM', t('set.wasmSub')],
  ];
  return (
    <div className="space-y-7">
      <div>
        <h3 className={h3}>{t('set.backend')}</h3>
        <p className="text-small text-muted-foreground mt-1">{t('set.backendSub')}</p>
        <div role="radiogroup" aria-label={t('set.backend')} className="grid grid-cols-3 gap-2 mt-3">
          {cards.map(([v, name, sub]) => (
            <button key={v} type="button" role="radio" aria-checked={settings.backend === v} disabled={busy || (v === 'webgpu' && env && !env.webgpu)} onClick={() => pick(v)}
              className="text-left rounded-xl border p-3 hover:bg-muted aria-checked:border-primary aria-checked:bg-accent disabled:opacity-45 disabled:cursor-not-allowed">
              <p className="text-small font-semibold">{name} {v === 'auto' && <span className="text-caption text-primary">{t('set.recommended')}</span>}</p>
              <p className="text-caption text-muted-foreground mt-0.5">{v === 'webgpu' && env && !env.webgpu ? t('set.noWebgpu') : sub}</p>
            </button>
          ))}
        </div>
        {busy && <p className="text-caption text-muted-foreground mt-2">{t('set.busy')}</p>}
        {env && (
          <dl className="grid grid-cols-[120px_1fr] gap-y-1.5 mt-4 text-small">
            <dt className="text-muted-foreground">WebGPU</dt><dd>{env.webgpu ? t('set.supported') + (env.shaderF16 ? ' · shader-f16' : '') : t('set.unsupported')}</dd>
            <dt className="text-muted-foreground">{t('set.threads')}</dt><dd>{env.crossOriginIsolated ? t('set.multi', { n: env.wasmThreads }) : t('set.single')}</dd>
          </dl>
        )}
      </div>
      <div>
        <h3 className={cn(h3, 'mb-3')}>{t('set.lang')}</h3>
        <Segmented size="md" label={t('set.lang')} value={settings.lang} onChange={(lang) => setS({ lang })} items={[{ value: 'ko', label: '한국어' }, { value: 'en', label: 'English' }]} />
      </div>
      <div>
        <h3 className={cn(h3, 'mb-3')}>{t('set.theme')}</h3>
        <Segmented size="md" label={t('set.theme')} value={settings.theme} onChange={(theme) => setS({ theme })}
          items={[{ value: 'system', label: t('set.system') }, { value: 'light', label: t('set.light') }, { value: 'dark', label: t('set.dark') }]} />
      </div>
      <SwitchRow label={t('set.remember')} desc={t('set.rememberSub')} checked={settings.remember} onChange={(remember) => setS({ remember })} />
    </div>
  );
}

function Storage() {
  const t = useT();
  const est = useApp((s) => s.cache);
  const entries = est?.entries;
  const baseUrl = useApp(selModelBaseUrl);
  const busy = useApp(selBusy);
  const [url, setUrl] = useState(baseUrl);
  const [verifying, setVerifying] = useState<string>();
  const [importing, setImporting] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const verify = async (id: string) => { setVerifying(id); await verifyCacheItem(id); setVerifying(undefined); };
  const doImport = async (f: File[]) => { setImporting(true); await importCache(f); setImporting(false); };
  const pub = entries?.filter((e) => e.kind === 'public') ?? [];
  const pubBytes = pub.reduce((a, e) => a + e.bytes, 0);
  const mineBytes = (entries ?? []).filter((e) => e.kind === 'my-model').reduce((a, e) => a + e.bytes, 0);
  const pct = (b: number) => (est?.quota ? `${Math.max(0.5, (b / est.quota) * 100)}%` : '0%');
  const applyUrl = (v: string) => { setUrl(v); setModelBaseUrl(v); };
  return (
    <div className="space-y-7">
      <div>
        <h3 className={h3}>{t('set.storage')}</h3>
        <div className="mt-3 rounded-xl border p-4">
          <div className="flex justify-between text-small tnum"><span className="font-semibold">{t('set.used', { n: fmtBytes(est?.usage ?? 0) })}</span><span className="text-muted-foreground">{t('set.quota', { n: fmtBytes(est?.quota ?? 0) })}</span></div>
          <div className="mt-2 h-2.5 rounded-full bg-muted overflow-hidden flex">
            <span className="h-full bg-primary" style={{ width: pct(pubBytes) }} /><span className="h-full bg-stem-vocal" style={{ width: pct(mineBytes) }} />
          </div>
          <div className="flex gap-4 mt-2 text-caption text-muted-foreground tnum">
            <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-primary" />{t('set.pubModels', { n: fmtBytes(pubBytes) })}</span>
            <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-stem-vocal" />{t('set.myModels', { n: fmtBytes(mineBytes) })}</span>
          </div>
          <p className="text-caption text-muted-foreground mt-3 flex items-center gap-1.5"><Lock className="size-3.5" />{est?.persisted ? t('set.persisted') : t('set.notPersisted')}</p>
        </div>
      </div>
      <div>
        <div className="flex items-center justify-between">
          <h3 className={h3}>{t('set.pubTitle')}</h3>
          <Button variant="ghost" className="h-8 text-destructive hover:text-destructive" disabled={!pub.length || busy} onClick={() => void clearCache()}>{t('set.clearAll')}</Button>
        </div>
        {pub.length ? (
          <ul className="mt-2 divide-y border-y">
            {pub.map((e) => (
              <li key={e.id} className="flex items-center gap-3 py-2.5 text-small">
                <span className="flex-1 min-w-0 truncate"><b>{t(roleOf(e.id))}</b> <span className="text-muted-foreground">{e.name}</span></span>
                <StatusChip status={verifying === e.id ? 'checking' : e.status} />
                <span className="tnum text-muted-foreground">{fmtBytes(e.bytes)}</span>
                <Button variant="ghost" className="h-8 px-2 rounded-lg text-small" disabled={busy || !!verifying} onClick={() => void verify(e.id)}>{t('cache.verify')}</Button>
                <button type="button" disabled={busy} className="grid place-items-center size-8 rounded-lg hover:bg-muted text-muted-foreground hover:text-destructive disabled:opacity-40" aria-label={t('common.delete')} onClick={() => void removeCacheItem(e.id)}><Trash2 className="size-4" /></button>
              </li>
            ))}
          </ul>
        ) : <p className="text-small text-muted-foreground mt-2">{entries ? t('set.pubEmpty') : t('env.checking')}</p>}
        <p className="text-caption text-muted-foreground mt-2">{t('set.integrityNote')}</p>
        <input ref={fileRef} type="file" multiple hidden onChange={(ev) => { const f = [...(ev.target.files ?? [])]; ev.target.value = ''; if (f.length) void doImport(f); }} />
        <Button variant="secondary" className="mt-3 h-9 rounded-lg" disabled={busy || importing} onClick={() => fileRef.current?.click()}>{importing ? <Loader2 className="animate-spin" /> : <FolderInput />}{t('set.fillLocal')}</Button>
        <p className="text-caption text-muted-foreground mt-1">{t('set.fillLocalSub')}</p>
      </div>
      <div>
        <h3 className={h3}>{t('set.repo')}</h3>
        <p className="text-small text-muted-foreground mt-1">{t('set.repoSub')}</p>
        <div className="flex gap-2 mt-3">
          <input value={url} onChange={(e) => setUrl(e.target.value)} aria-label={t('set.repo')} spellCheck={false}
            className="flex-1 min-w-0 h-9 rounded-lg border border-input bg-card px-3 text-small font-mono focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" />
          <Button className="h-9 rounded-lg" disabled={busy || url === baseUrl || !url.trim()} onClick={() => applyUrl(url.trim())}>{t('common.apply')}</Button>
          <Button variant="secondary" className="h-9 rounded-lg" disabled={busy || baseUrl === DEFAULT_BASE_URL} onClick={() => applyUrl(DEFAULT_BASE_URL)}>{t('set.repoDefault')}</Button>
        </div>
        <p className="text-caption text-muted-foreground mt-2">{t('set.repoCsp')}</p>
      </div>
    </div>
  );
}

function Models() {
  const t = useT();
  const busy = useApp(selBusy);
  const lang = useApp((s) => s.settings.lang);
  const entries = useApp((s) => s.cache?.entries);
  const list = useApp((s) => s.myModels);
  const [editing, setEditing] = useState<string>();
  return (
    <div>
      <h3 className={h3}>{t('set.mineTitle')}</h3>
      <p className="text-small text-muted-foreground mt-1">{t('set.mineSub')}</p>
      {list?.length ? (
        <ul className="mt-4 divide-y border-y">
          {list.map((m) => (
            <li key={m.id} className="flex items-center gap-3 py-3">
              <span className="grid place-items-center size-9 rounded-lg bg-stem-vocal/15 text-stem-vocal shrink-0"><MicVocal className="size-4" /></span>
              <div className="flex-1 min-w-0">
                {editing === m.id
                  ? <RenameInput name={m.name} onDone={async (name) => { setEditing(undefined); if (name && name !== m.name) await renameMine(m.id, name); }} />
                  : <p className="text-small font-semibold flex items-center gap-1 min-w-0"><span className="truncate">{m.name}</span>
                      <button type="button" disabled={busy} className="grid place-items-center size-6 shrink-0 rounded-md text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-40" aria-label={t('mine.rename')} title={t('mine.rename')} onClick={() => setEditing(m.id)}><Pencil className="size-3.5" /></button>
                    </p>}
                <p className="text-caption text-muted-foreground">{[m.source === 'onnx' && 'ONNX', m.version, fmtHz(m.sr), m.f0 ? t('model.f0') : t('model.nof0'), m.indexId ? (m.index ? t('mine.indexN', { n: m.index.nVectors.toLocaleString() }) : 'index') : t('set.noIndex'), (m.lastUsed || m.savedAt) > 0 && t('mine.lastUsed', { d: fmtDate(m.lastUsed || m.savedAt, lang) })].filter(Boolean).join(' · ')}</p>
              </div>
              <span className="text-caption text-muted-foreground tnum">{fmtBytes(entries?.find((e) => e.id === m.id)?.bytes ?? m.bytes)}</span>
              <Button variant="secondary" className="h-8 rounded-lg" disabled={busy} onClick={() => void openMyModel(m.id)}>{t('set.use')}</Button>
              <button type="button" disabled={busy} className="grid place-items-center size-8 rounded-lg hover:bg-muted text-muted-foreground hover:text-destructive disabled:opacity-40" aria-label={t('common.delete')} onClick={() => void removeMine(m.id, m.name)}><Trash2 className="size-4" /></button>
            </li>
          ))}
        </ul>
      ) : <p className="mt-4 rounded-xl bg-muted/60 p-4 text-small text-muted-foreground">{list ? t('mine.empty') : t('env.checking')}</p>}
      <p className="text-caption text-muted-foreground mt-3">{t('set.mineDeleteNote')}</p>
    </div>
  );
}

function Privacy() {
  const t = useT();
  const baseUrl = useApp(selModelBaseUrl);
  return (
    <div className="space-y-7">
      <div>
        <h3 className={h3}>{t('privacy.chip')}</h3>
        <p className="text-small text-muted-foreground mt-1">{t('privacy.body')}</p>
        <table className="w-full mt-4 text-small">
          <thead className="text-left text-caption text-muted-foreground"><tr><th className="py-2 font-semibold">{t('privacy.addr')}</th><th className="font-semibold">{t('privacy.what')}</th><th className="font-semibold">{t('privacy.when')}</th></tr></thead>
          <tbody className="divide-y border-y">
            <tr><td className="py-2 font-mono text-caption">{location.host}</td><td>{t('privacy.app')}</td><td>{t('privacy.appWhen')}</td></tr>
            <tr><td className="py-2 font-mono text-caption">{hostOf(baseUrl)}</td><td>{t('privacy.models')}</td><td>{t('privacy.modelsWhen')}</td></tr>
          </tbody>
        </table>
      </div>
      <div>
        <h3 className={h3}>{t('privacy.noticeTitle')}</h3>
        <p className="text-small text-muted-foreground mt-1">{t('notice.body')}</p>
        <Button variant="secondary" className="mt-3 h-9 rounded-lg" onClick={showNoticeAgain}>{t('privacy.showNotice')}</Button>
      </div>
    </div>
  );
}

function About() {
  const t = useT();
  const env = useApp((s) => s.env);
  const baseUrl = useApp(selModelBaseUrl);
  return (
    <div className="space-y-6">
      <h3 className={h3}>Web RVC</h3>
      <dl className="grid grid-cols-[140px_1fr] gap-y-2 text-small">
        <dt className="text-muted-foreground">{t('about.version')}</dt><dd className="tnum">{version}</dd>
        <dt className="text-muted-foreground">{t('about.modelRepo')}</dt><dd className="font-mono text-caption break-all">{baseUrl}</dd>
        {env && <><dt className="text-muted-foreground">{t('about.revision')}</dt><dd className="font-mono text-caption break-all">{env.modelRevision}</dd></>}
        {env && <><dt className="text-muted-foreground">GPU</dt><dd className="text-caption break-all">{env.gpuAdapter ?? (env.webgpu ? '-' : t('set.unsupported'))}</dd></>}
      </dl>
      <div>
        <h3 className={h3}>{t('about.licenses')}</h3>
        <ul className="mt-2 text-small text-muted-foreground space-y-1">
          {['onnxruntime-web · MIT', 'React, Radix UI, shadcn/ui, lucide · MIT/ISC', 'wavesurfer.js · BSD-3-Clause', 'mediabunny · MPL-2.0', '@mediabunny/mp3-encoder (LAME) · LGPL-2.1', 'Pretendard · SIL OFL 1.1', 'RVC, UVR5 (models) · MIT', 'ContentVec / HuBERT, RMVPE · ' + t('about.modelLic')].map((x) => <li key={x}>{x}</li>)}
        </ul>
      </div>
      <div>
        <h3 className={h3}>{t('about.lameTitle')}</h3>
        <p className="text-small text-muted-foreground mt-1">{t('about.lame')}</p>
        <p className="text-caption mt-1.5 flex flex-wrap gap-x-3">
          <a className="text-primary underline-offset-2 hover:underline" href="https://lame.sourceforge.io/" target="_blank" rel="noreferrer">lame.sourceforge.io</a>
          <a className="text-primary underline-offset-2 hover:underline" href="https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html" target="_blank" rel="noreferrer">LGPL-2.1</a>
          <a className="text-primary underline-offset-2 hover:underline" href="https://github.com/Vanilagy/mediabunny" target="_blank" rel="noreferrer">mediabunny</a>
        </p>
      </div>
      <Button variant="secondary" className="h-9 rounded-lg" onClick={copyEnvInfo}><Clipboard />{t('about.copyEnv')}</Button>
      <p className="text-caption text-muted-foreground">{t('about.copyNote')}</p>
    </div>
  );
}

const TABS: [SettingsTab, typeof Info][] = [['general', SlidersHorizontal], ['storage', HardDrive], ['models', FolderHeart], ['privacy', ShieldCheck], ['about', Info]];

export function SettingsDialog() {
  const t = useT();
  const tab = useApp((s) => s.settingsTab);
  // DESIGN S5: #settings/<tab> opens a tab directly.
  useEffect(() => {
    const read = () => { const m = /^#settings(?:\/(\w+))?/.exec(location.hash); if (m) openSettings(TABS.find(([k]) => k === m[1])?.[0] ?? 'general'); };
    read();
    addEventListener('hashchange', read);
    return () => removeEventListener('hashchange', read);
  }, []);
  useEffect(() => {
    const want = tab ? `#settings/${tab}` : '';
    if (location.hash !== want && (tab || location.hash.startsWith('#settings'))) history.replaceState(null, '', want || location.pathname + location.search);
  }, [tab]);
  return (
    <D.Root open={!!tab} onOpenChange={(o) => !o && openSettings(undefined)}>
      <D.Portal>
        <D.Overlay className="fixed inset-0 z-50 bg-overlay/50 data-[state=open]:animate-in data-[state=open]:fade-in-0" />
        <D.Content className="fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2 w-[min(860px,calc(100%-2rem))] h-[min(600px,calc(100%-2rem))] rounded-2xl bg-card shadow-pop border overflow-hidden data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95" aria-describedby={undefined}>
          <Tabs.Root value={tab ?? 'general'} onValueChange={(v) => openSettings(v as SettingsTab)} orientation="vertical" className="flex h-full">
            <Tabs.List className="w-52 shrink-0 border-r bg-muted/40 p-3 flex flex-col gap-1" aria-label={t('settings.title')}>
              <D.Title className="px-3 py-2 text-h3">{t('settings.title')}</D.Title>
              {TABS.map(([k, Icon]) => (
                <Tabs.Trigger key={k} value={k} className="flex items-center gap-2 h-9 px-3 rounded-lg text-small font-semibold text-muted-foreground hover:bg-muted hover:text-foreground data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-card focus-visible:outline-2 focus-visible:outline-ring">
                  <Icon className="size-4" />{t(`settings.${k}`)}
                </Tabs.Trigger>
              ))}
            </Tabs.List>
            <div className="flex-1 min-w-0 relative">
              <D.Close className="absolute right-4 top-4 grid place-items-center size-8 rounded-lg hover:bg-muted z-10" aria-label={t('common.close')}><X className="size-4" /></D.Close>
              {(['general', 'storage', 'models', 'privacy', 'about'] as const).map((k) => (
                <Tabs.Content key={k} value={k} className="h-full overflow-y-auto p-7 pr-14 outline-none">
                  {k === 'general' ? <General /> : k === 'storage' ? <Storage /> : k === 'models' ? <Models /> : k === 'privacy' ? <Privacy /> : <About />}
                </Tabs.Content>
              ))}
            </div>
          </Tabs.Root>
        </D.Content>
      </D.Portal>
    </D.Root>
  );
}
