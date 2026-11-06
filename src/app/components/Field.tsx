// Design-system form pieces: HelpTip, SliderField, Segmented, SwitchRow, SectionLabel.
import { useId, useState, type ReactNode } from 'react';
import { CircleHelp, RotateCcw, TriangleAlert } from 'lucide-react';
import { RadioGroup as RG } from 'radix-ui';
import { Slider } from './ui/slider';
import { Switch } from './ui/switch';
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip';
import { cn } from '../lib/utils';
import { useT } from '../i18n';

export function HelpTip({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" className="text-muted-foreground hover:text-foreground rounded-full focus-visible:outline-2" aria-label={text}>
          <CircleHelp className="size-3.5" strokeWidth={1.75} />
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-[280px] text-caption">{text}</TooltipContent>
    </Tooltip>
  );
}

const round = (v: number, step: number) => Math.round(v / step) * step;
const decimals = (step: number) => (String(step).split('.')[1] ?? '').length;

interface SliderFieldProps {
  label: ReactNode;
  name?: string; // accessible name when label is not a plain string
  help?: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  unit?: string;
  defaultValue?: number;
  disabled?: boolean;
  note?: ReactNode; // reason when disabled, or helper text
  warning?: ReactNode;
  chips?: { label: string; value: number }[];
  valueText?: (v: number) => string;
  badge?: ReactNode;
  dot?: string; // stem colour class for mixer labels
}

/** DESIGN SliderField: label + help, number input (snaps on blur/Enter), slider, chips, reset to default. */
export function SliderField(p: SliderFieldProps) {
  const t = useT();
  const id = useId();
  const fmt = (v: number) => v.toFixed(decimals(p.step));
  const [draft, setDraft] = useState<string>(); // typed text until blur/Enter; otherwise the value is shown
  const text = draft ?? fmt(p.value);
  const commit = () => {
    setDraft(undefined);
    const n = Number(text.replace('−', '-').replace('+', ''));
    if (!Number.isFinite(n) || draft === undefined) return;
    const v = Math.min(p.max, Math.max(p.min, round(n, p.step)));
    if (v !== p.value) p.onChange(+v.toFixed(decimals(p.step)));
  };
  const changed = p.defaultValue !== undefined && p.value !== p.defaultValue && !p.disabled;
  return (
    <div>
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-1 min-w-0">
          {p.dot && <span className={cn('size-2 rounded-full shrink-0', p.dot)} />}
          <label htmlFor={id} className="text-small font-semibold">{p.label}</label>
          {p.help && <HelpTip text={p.help} />}
          {p.badge}
          {changed && (
            <button type="button" onClick={() => p.onChange(p.defaultValue!)} className="grid place-items-center size-6 rounded-md text-muted-foreground hover:bg-muted hover:text-foreground" aria-label={t('field.reset')} title={t('field.reset')}>
              <RotateCcw className="size-3.5" />
            </button>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <input
            id={id}
            inputMode="decimal"
            value={p.disabled ? '–' : text}
            disabled={p.disabled}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => e.key === 'Enter' && commit()}
            className={cn('w-16 h-8 rounded-md border bg-card text-right px-2 text-small tnum focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:bg-muted disabled:opacity-45', p.warning ? 'border-warning' : 'border-input')}
          />
          {p.unit && <span className="text-caption text-muted-foreground min-w-6">{p.unit}</span>}
        </div>
      </div>
      <Slider
        value={[p.value]}
        min={p.min}
        max={p.max}
        step={p.step}
        disabled={p.disabled}
        onValueChange={([v]) => p.onChange(+v.toFixed(decimals(p.step)))}
        aria-label={p.name ?? (typeof p.label === 'string' ? p.label : undefined)}
        aria-valuetext={p.valueText?.(p.value) ?? `${fmt(p.value)}${p.unit ? ` ${p.unit}` : ''}`}
        className="py-2"
      />
      {p.chips && !p.disabled && (
        <div className="flex flex-wrap gap-1.5 mt-2">
          {p.chips.map((c) => (
            <button key={c.label} type="button" onClick={() => p.onChange(c.value)} aria-pressed={p.value === c.value}
              className="h-7 px-2.5 rounded-full bg-muted text-caption font-semibold hover:bg-secondary aria-pressed:bg-accent aria-pressed:text-accent-foreground">
              {c.label}
            </button>
          ))}
        </div>
      )}
      {p.warning && <p className="mt-2 flex gap-1.5 text-caption text-warning"><TriangleAlert className="size-3.5 shrink-0 mt-px" />{p.warning}</p>}
      {p.note && <p className="text-caption text-muted-foreground mt-1.5">{p.note}</p>}
    </div>
  );
}

interface SegmentedProps<T extends string> {
  value: T | undefined; // undefined = nothing chosen yet
  onChange: (v: T) => void;
  items: { value: T; label: ReactNode; disabled?: boolean }[];
  label: string;
  size?: 'sm' | 'md';
  className?: string;
}

/** DESIGN SegmentedControl as a radio group: arrow keys move and select (QA B9, role=radio convention). */
export function Segmented<T extends string>({ value, onChange, items, label, size = 'sm', className }: SegmentedProps<T>) {
  return (
    <RG.Root value={value ?? ''} onValueChange={(v) => onChange(v as T)} aria-label={label} orientation="horizontal"
      className={cn('inline-grid grid-flow-col auto-cols-fr p-1 rounded-xl bg-muted', size === 'sm' && 'p-0.5 rounded-lg', className)}>
      {items.map((it) => (
        <RG.Item key={it.value} value={it.value} disabled={it.disabled}
          className={cn('rounded-lg text-muted-foreground font-semibold transition-colors hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed data-[state=checked]:bg-card data-[state=checked]:text-foreground data-[state=checked]:shadow-card focus-visible:outline-2 focus-visible:outline-ring',
            size === 'sm' ? 'h-7 px-2.5 text-caption rounded-md' : 'h-10 px-3 text-small')}>
          {it.label}
        </RG.Item>
      ))}
    </RG.Root>
  );
}

export function SwitchRow({ label, desc, checked, onChange, disabled }: { label: ReactNode; desc?: ReactNode; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  const id = useId();
  return (
    <div className="flex items-center gap-3">
      <label htmlFor={id} className="flex-1 min-w-0 cursor-pointer">
        <span className="text-small font-semibold">{label}</span>
        {desc && <span className="block text-caption text-muted-foreground mt-0.5">{desc}</span>}
      </label>
      <Switch id={id} checked={checked} onCheckedChange={onChange} disabled={disabled} />
    </div>
  );
}

export const SectionLabel = ({ children }: { children: ReactNode }) => (
  <p className="text-caption font-bold text-muted-foreground uppercase tracking-wide mb-3">{children}</p>
);

/** Number + step card header: (1) 노래 */
export function StepCard({ n, title, aside, children, className, id }: { n: number; title: string; aside?: ReactNode; children: ReactNode; className?: string; id?: string }) {
  return (
    <section className={cn('rounded-2xl bg-card shadow-card p-6 min-w-0', className)} aria-labelledby={id}>
      <div className="flex items-center gap-2.5 mb-4 min-h-8">
        <span className="grid place-items-center size-6 rounded-full bg-foreground text-background text-caption font-bold">{n}</span>
        <h2 id={id} className="text-h2">{title}</h2>
        <span className="flex-1" />
        {aside}
      </div>
      {children}
    </section>
  );
}
