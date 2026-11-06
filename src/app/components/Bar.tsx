// Progress bar: `value` 0..100, or indeterminate (loading / verifying) when undefined.
import { cn } from '../lib/utils';

export function Bar({ value, label, className }: { value?: number; label?: string; className?: string }) {
  const pct = value === undefined ? undefined : Math.max(0, Math.min(100, value));
  return (
    <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct === undefined ? undefined : Math.floor(pct)}
      className={cn('h-1.5 rounded-full bg-muted overflow-hidden', className)}>
      {pct === undefined
        ? <div className="h-full w-2/5 rounded-full bg-primary animate-indet" />
        : <div className="h-full rounded-full bg-primary transition-[width] duration-300 ease-linear" style={{ width: `${pct}%` }} />}
    </div>
  );
}
