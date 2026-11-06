// AppHeader + EnvBadge, S6 banners, ET-2 first-run notice, mobile gate.
import { useEffect, useState } from 'react';
import { AudioWaveform, Info, Loader2, MonitorSmartphone, Settings, ShieldCheck } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from './ui/button';
import { Banner, NoticeButton } from './Notice';
import { cn } from '../lib/utils';
import { fmtBytes } from '../lib/format';
import { ackNotice, dismiss, initEngine, openSettings, shortenToClip } from '../controller';
import { selMissingBytes, useApp } from '../store';
import { useShallow } from 'zustand/react/shallow';
import { useT } from '../i18n';

export function EnvBadge() {
  const t = useT();
  const { env, envError, online, engineReady } = useApp(useShallow((s) => ({ env: s.env, envError: s.envError, online: s.online, engineReady: s.engineReady })));
  const fallback = useApp((s) => s.job?.fallback);
  let tone = 'muted', text = t('env.checking'), spin = false, help = t('env.badgeTitle');
  if (!online && engineReady) text = t('env.offline');
  else if (envError) { tone = 'error'; text = t('env.error'); }
  else if (!env || !engineReady) spin = true;
  else if (env.backend === 'wasm' || fallback) { tone = 'warning'; text = env.crossOriginIsolated ? t('env.wasm') : t('env.wasmSingle'); help = env.crossOriginIsolated ? t('env.wasmHelp') : t('env.wasmSingleHelp'); }
  else { tone = 'success'; text = t('env.webgpu'); help = t('env.webgpuHelp'); }
  return (
    <button type="button" onClick={() => openSettings('general')} title={help}
      className={cn('inline-flex items-center gap-1.5 h-8 px-3 rounded-full text-small font-semibold',
        tone === 'success' && 'bg-success-soft text-success', tone === 'warning' && 'bg-warning-soft text-warning',
        tone === 'error' && 'bg-destructive-soft text-destructive', tone === 'muted' && 'bg-muted text-muted-foreground')}>
      {spin ? <Loader2 className="size-3.5 animate-spin" /> : <span className={cn('size-2 rounded-full', tone === 'muted' ? 'bg-muted-foreground' : 'bg-current')} />}
      {text}
    </button>
  );
}

export function AppHeader() {
  const t = useT();
  return (
    <header className="sticky top-0 z-40 border-b bg-card/85 backdrop-blur">
      <div className="mx-auto max-w-[1280px] h-14 px-6 flex items-center gap-3">
        <div className="flex items-center gap-2 font-bold text-lead">
          <span className="grid place-items-center size-7 rounded-lg bg-primary text-primary-foreground"><AudioWaveform className="size-4" /></span>
          Web RVC
        </div>
        <span className="flex-1" />
        <button type="button" onClick={() => openSettings('privacy')} title={t('privacy.chipTitle')}
          className="hidden md:inline-flex items-center gap-1.5 h-8 px-3 rounded-full text-small text-muted-foreground hover:bg-muted">
          <ShieldCheck className="size-4 text-success" />{t('privacy.chip')}
        </button>
        <EnvBadge />
        <button type="button" id="settings-btn" onClick={() => openSettings('general')} className="grid place-items-center size-9 rounded-lg hover:bg-muted" aria-label={t('settings.title')}>
          <Settings className="size-5" strokeWidth={1.75} />
        </button>
        <a href="https://github.com/ouor/rvc-webgpu" target="_blank" rel="noopener noreferrer" className="grid place-items-center size-9 rounded-lg hover:bg-muted"
          aria-label={t('header.github')} title={t('header.github')}>
          {/* Octicons mark-github (MIT); lucide has no brand icons */}
          <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" className="size-5">
            <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z" />
          </svg>
        </a>
      </div>
    </header>
  );
}

/** S6 banners: never block, closable for the session. */
export function Banners() {
  const t = useT();
  const { env, envError, dismissed, crashed } = useApp(useShallow((s) => ({ env: s.env, envError: s.envError, dismissed: s.dismissed, crashed: s.crashed })));
  const need = useApp(selMissingBytes);
  const shorten = shortenToClip;
  const items = [];
  if (envError) items.push(
    <Banner key="init" tone="error" title={t('banner.initTitle')} body={t('banner.initBody')}
      actions={<><NoticeButton onClick={() => void initEngine()}>{t('common.retry')}</NoticeButton><NoticeButton onClick={() => openSettings('storage')}>{t('banner.repoSettings')}</NoticeButton></>} />,
  );
  if (crashed && !dismissed.includes('crash')) items.push(
    <Banner key="crash" tone="error" title={t('banner.crashTitle')} body={t('banner.crashBody')}
      actions={<NoticeButton onClick={shorten}>{t('banner.shorten')}</NoticeButton>} onClose={() => dismiss('crash')} />,
  );
  if (env?.backend === 'wasm' && !dismissed.includes('wasm')) items.push(
    <Banner key="wasm" tone="warning" title={env.crossOriginIsolated ? t('banner.wasmTitle') : t('banner.singleTitle')} body={env.crossOriginIsolated ? t('banner.wasmBody') : t('banner.singleBody')}
      actions={<><NoticeButton onClick={shorten}>{t('banner.shorten')}</NoticeButton><NoticeButton onClick={() => openSettings('general')}>{t('common.more')}</NoticeButton></>} onClose={() => dismiss('wasm')} />,
  );
  const free = env ? env.storage.quota - env.storage.usage : Infinity;
  if (env && env.storage.quota > 0 && need > free && !dismissed.includes('quota')) items.push(
    <Banner key="quota" tone="warning" title={t('banner.quotaTitle')} body={t('banner.quotaBody', { need: fmtBytes(need), free: fmtBytes(Math.max(0, free)) })}
      actions={<NoticeButton onClick={() => openSettings('storage')}>{t('banner.manage')}</NoticeButton>} onClose={() => dismiss('quota')} />,
  );
  const mem = (navigator as { deviceMemory?: number }).deviceMemory;
  if (mem && mem < 8 && !dismissed.includes('ram')) items.push(
    <Banner key="ram" tone="warning" title={t('banner.ramTitle', { gb: mem })} body={t('banner.ramBody')}
      actions={<NoticeButton onClick={shorten}>{t('banner.shorten')}</NoticeButton>} onClose={() => dismiss('ram')} />,
  );
  if (!items.length) return null;
  return <div className="mx-auto max-w-[1280px] px-6 pt-5 space-y-3">{items}</div>;
}

export function FirstRunNotice() {
  const t = useT();
  const ack = useApp((s) => s.settings.noticeAck);
  if (ack) return null;
  return (
    <div className="mb-5 flex items-start gap-3 rounded-xl bg-card shadow-card px-5 py-4">
      <Info className="size-5 text-accent-foreground mt-0.5 shrink-0" />
      <div className="flex-1">
        <p className="font-semibold">{t('notice.title')}</p>
        <p className="text-small mt-0.5">{t('notice.body')}</p>
      </div>
      <Button variant="secondary" onClick={ackNotice}>{t('notice.ack')}</Button>
    </div>
  );
}

const NARROW = '(max-width: 767px)';
const PHONE = /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent) && navigator.maxTouchPoints > 0; // a device does not change

/** QA B15: follows window resizes (a phone stays a phone). */
function useMobile() {
  const [narrow, setNarrow] = useState(() => matchMedia(NARROW).matches);
  useEffect(() => {
    const mq = matchMedia(NARROW);
    const on = () => setNarrow(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return narrow || PHONE;
}

export function MobileGate() {
  const t = useT();
  const [go, setGo] = useState(() => { try { return sessionStorage.getItem('web-rvc:mobile-ok') === '1'; } catch { return false; } });
  const mobile = useMobile();
  if (go || !mobile) return null;
  return (
    <div className="fixed inset-0 z-50 bg-background grid place-items-center p-6">
      <div className="max-w-[360px] w-full text-center">
        <span className="mx-auto grid place-items-center size-14 rounded-2xl bg-warning-soft text-warning"><MonitorSmartphone className="size-7" /></span>
        <h1 className="text-h1 mt-5">{t('mobile.title')}</h1>
        <p className="text-muted-foreground mt-2">{t('mobile.body')}</p>
        <div className="mt-6 rounded-xl bg-card shadow-card p-4 text-left text-small">
          <p className="font-semibold">{t('mobile.recTitle')}</p>
          <p className="text-muted-foreground mt-1">{t('mobile.rec')}</p>
        </div>
        <Button size="cta" className="mt-6 w-full" onClick={() => { void navigator.clipboard.writeText(location.href); toast(t('toast.linkCopied')); }}>{t('mobile.copy')}</Button>
        <Button variant="ghost" size="cta" className="mt-2 w-full text-muted-foreground" onClick={() => { try { sessionStorage.setItem('web-rvc:mobile-ok', '1'); } catch { /* ignore */ } setGo(true); }}>{t('mobile.continue')}</Button>
      </div>
    </div>
  );
}
