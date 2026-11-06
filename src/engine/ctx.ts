// Execution context passed to every stage (ARCHITECTURE §6.3).
import { AppError } from '../shared/errors';
import type { Backend, OnProgress, Progress } from '../shared/types';
import type { Prng } from '../dsp/prng';

export interface RunCtx {
  backend: Backend;
  progress(done: number, total: number): void;
  throwIfCancelled(): void;
  rng: Prng;
}

export class CancelFlag {
  private cancelled = false;
  reset() { this.cancelled = false; }
  cancel() { this.cancelled = true; }
  check() { if (this.cancelled) throw new AppError('CANCELLED', 'cancelled by user'); }
}

/**
 * At most one progress event per `ms` within a step (QA B1: one event per 64 KB download read froze the UI).
 * A change of stage, phase, file or item, a finished step and reuse notices pass at once; the last dropped
 * event is delivered after `ms`, so the UI never stops short.
 */
export function throttleProgress(on: OnProgress, ms = 100): OnProgress {
  let last: Progress | undefined, at = -Infinity, pending: Progress | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const emit = (p: Progress) => {
    if (timer) { clearTimeout(timer); timer = undefined; }
    pending = undefined; last = p; at = performance.now();
    on(p);
  };
  return (p) => {
    const step = !last || p.stage !== last.stage || p.phase !== last.phase || p.file !== last.file || p.item !== last.item;
    if (step || p.done >= p.total || p.reused || performance.now() - at >= ms) return emit(p);
    pending = p;
    timer ??= setTimeout(() => { timer = undefined; if (pending) emit(pending); }, ms - (performance.now() - at));
  };
}

/** Gives the event loop a turn so cancel() and other RPCs can be processed between CPU-heavy steps. */
export const yieldToLoop = () => new Promise<void>((r) => setTimeout(r, 0));
