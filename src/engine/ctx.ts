// Execution context passed to every stage (ARCHITECTURE §6.3).
import { AppError } from '../shared/errors';
import type { Backend } from '../shared/types';
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

/** Gives the event loop a turn so cancel() and other RPCs can be processed between CPU-heavy steps. */
export const yieldToLoop = () => new Promise<void>((r) => setTimeout(r, 0));
