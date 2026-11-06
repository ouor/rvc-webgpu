// InlineError (EX-8: what happened / what to do / copyable detail) and Banner (S6).
import { useEffect, useState, type ReactNode } from 'react';
import { Check, CircleX, Copy, Info, ShieldAlert, TriangleAlert, X } from 'lucide-react';
import { toast } from 'sonner';
import type { AppError } from '@/shared/errors';
import { Button } from './ui/button';
import { cn } from '../lib/utils';
import { errText, useT } from '../i18n';

type Tone = 'error' | 'warning' | 'info';
const toneCls: Record<Tone, string> = {
  error: 'border-destructive/40 bg-destructive-soft',
  warning: 'border-warning/40 bg-warning-soft',
  info: 'border-transparent bg-accent',
};
const toneIcon: Record<Tone, string> = { error: 'text-destructive', warning: 'text-warning', info: 'text-accent-foreground' };

/** DESIGN 9.10: title, what to do, the message to send (copy button, no toast), details (longer explanation, code, copyable report). */
export function InlineError({ error, tone = 'error', file, title, body, actions, className }: {
  error?: AppError; tone?: Tone; file?: string; title?: ReactNode; body?: ReactNode; actions?: ReactNode; className?: string;
}) {
  const t = useT();
  const txt = error ? errText(error, file) : undefined;
  const Icon = error?.code === 'PTH_UNSAFE' ? ShieldAlert : tone === 'error' ? CircleX : tone === 'warning' ? TriangleAlert : Info;
  const detail = error && [`${error.code}: ${error.message}`, file && `file=${file}`, error.detail, `ua=${navigator.userAgent}`].filter(Boolean).join('\n');
  const lead = body ?? txt?.todo ?? txt?.body;
  return (
    <div className={cn('rounded-xl border p-4', toneCls[tone], className)} role={tone === 'error' ? 'alert' : 'status'}>
      <div className="flex items-start gap-3">
        <Icon className={cn('size-5 mt-0.5 shrink-0', toneIcon[tone])} strokeWidth={1.75} />
        <div className="flex-1 min-w-0">
          <p className={cn('font-semibold', tone !== 'error' && 'text-small')}>{title ?? txt?.title}</p>
          <p className="text-small mt-1">
            {file && <b className="break-all">{file}</b>}{file && ' · '}{lead}
          </p>
          {txt?.ask && <AskBox text={txt.ask} />}
          {detail && (
            <details className="mt-2 text-caption">
              <summary className="cursor-pointer text-muted-foreground hover:text-foreground">{t('common.more')}</summary>
              {txt?.todo && <p className="mt-2 text-small">{txt.body}</p>}
              <p className="mt-2 text-muted-foreground">{t('err.code')}: <span className="font-mono">{error.code}</span></p>
              <pre className="mt-1 p-2 rounded-md bg-card font-mono text-[11px] whitespace-pre-wrap break-all max-h-40 overflow-auto">{detail}</pre>
              <button type="button" className="mt-1 text-accent-foreground font-semibold" onClick={() => { void navigator.clipboard.writeText(detail); toast(t('toast.copied')); }}>{t('common.copy')}</button>
            </details>
          )}
          {actions && <div className="mt-3 flex flex-wrap gap-2">{actions}</div>}
        </div>
      </div>
    </div>
  );
}

/** "보낼 문장": only the sentence is copied (no file name, path or code); the button label confirms for 2 s. */
function AskBox({ text }: { text: string }) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  useEffect(() => { if (!copied) return; const id = setTimeout(() => setCopied(false), 2000); return () => clearTimeout(id); }, [copied]);
  return (
    <div className="mt-3 flex items-start gap-3 rounded-lg border bg-card p-3">
      <div className="flex-1 min-w-0">
        <p className="text-caption font-semibold text-muted-foreground">{t('err.askLabel')}</p>
        <p className="text-small mt-0.5">{text}</p>
      </div>
      <Button size="sm" variant="secondary" className="shrink-0" onClick={() => { void navigator.clipboard.writeText(text).then(() => setCopied(true)); }}>
        {copied ? <><Check />{t('err.copied')}</> : <><Copy />{t('err.copyAsk')}</>}
      </Button>
    </div>
  );
}

export function Banner({ tone, title, body, actions, onClose, icon }: { tone: Tone; title: ReactNode; body?: ReactNode; actions?: ReactNode; onClose?: () => void; icon?: ReactNode }) {
  const t = useT();
  const Icon = tone === 'error' ? CircleX : tone === 'warning' ? TriangleAlert : Info;
  return (
    <div role={tone === 'error' ? 'alert' : 'status'} className={cn('flex items-start gap-3 rounded-xl px-4 py-3', tone === 'error' ? 'bg-destructive-soft' : tone === 'warning' ? 'bg-warning-soft' : 'bg-accent')}>
      {icon ?? <Icon className={cn('size-5 mt-0.5 shrink-0', toneIcon[tone])} strokeWidth={1.75} />}
      <div className="flex-1 min-w-0">
        <p className="font-semibold">{title}</p>
        {body && <p className="text-small text-muted-foreground mt-0.5">{body}</p>}
      </div>
      {actions}
      {onClose && (
        <button type="button" onClick={onClose} className="grid place-items-center size-8 rounded-lg hover:bg-card shrink-0" aria-label={t('common.close')}>
          <X className="size-4" />
        </button>
      )}
    </div>
  );
}

/** Small action button used inside notices and banners. */
export const NoticeButton = ({ children, onClick, primary }: { children: ReactNode; onClick: () => void; primary?: boolean }) => (
  <Button size="sm" variant={primary ? 'default' : 'card'} onClick={onClick}>{children}</Button>
);
