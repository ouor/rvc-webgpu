// InlineError (EX-8: what happened / what to do / copyable detail) and Banner (S6).
import type { ReactNode } from 'react';
import { CircleX, Info, ShieldAlert, TriangleAlert, X } from 'lucide-react';
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
const toneIcon: Record<Tone, string> = { error: 'text-destructive', warning: 'text-warning', info: 'text-primary' };

export function InlineError({ error, tone = 'error', file, title, body, actions, className }: {
  error?: AppError; tone?: Tone; file?: string; title?: ReactNode; body?: ReactNode; actions?: ReactNode; className?: string;
}) {
  const t = useT();
  const txt = error ? errText(error) : undefined;
  const Icon = error?.code === 'PTH_UNSAFE' ? ShieldAlert : tone === 'error' ? CircleX : tone === 'warning' ? TriangleAlert : Info;
  const detail = error && [`${error.code}: ${error.message}`, file && `file=${file}`, error.detail, `ua=${navigator.userAgent}`].filter(Boolean).join('\n');
  return (
    <div className={cn('rounded-xl border p-4', toneCls[tone], className)} role={tone === 'error' ? 'alert' : 'status'}>
      <div className="flex items-start gap-3">
        <Icon className={cn('size-5 mt-0.5 shrink-0', toneIcon[tone])} strokeWidth={1.75} />
        <div className="flex-1 min-w-0">
          <p className={cn('font-semibold', tone !== 'error' && 'text-small')}>{title ?? txt?.title}</p>
          <p className="text-small text-muted-foreground mt-1">
            {file && <b className="text-foreground break-all">{file}</b>}{file && ' · '}{body ?? txt?.body}
          </p>
          {detail && (
            <details className="mt-2 text-caption">
              <summary className="cursor-pointer text-muted-foreground hover:text-foreground">{t('err.detail')}</summary>
              <pre className="mt-2 p-2 rounded-md bg-card font-mono text-[11px] whitespace-pre-wrap break-all max-h-40 overflow-auto">{detail}</pre>
              <button type="button" className="mt-1 text-primary font-semibold" onClick={() => { void navigator.clipboard.writeText(detail); toast(t('toast.copied')); }}>{t('common.copy')}</button>
            </details>
          )}
          {actions && <div className="mt-3 flex flex-wrap gap-2">{actions}</div>}
        </div>
      </div>
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
