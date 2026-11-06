// Comlink throw handler that keeps AppError (code, detail, recoverable) across the worker boundary (§10).
// Both sides call installErrorTransfer() once: the worker before expose(), the client before wrap().
import * as Comlink from 'comlink';
import { AppError, type ErrorCode } from './errors';

interface Wire { code: ErrorCode; message: string; detail?: string; recoverable: boolean; stack?: string }
type ThrowHandler = Comlink.TransferHandler<unknown, unknown>;

export function installErrorTransfer(): void {
  const prev = Comlink.transferHandlers.get('throw') as ThrowHandler;
  Comlink.transferHandlers.set('throw', {
    canHandle: prev.canHandle,
    serialize(v: unknown) {
      const e = (v as { value: unknown }).value;
      if (e instanceof AppError) {
        const w: Wire = { code: e.code, message: e.message, detail: e.detail, recoverable: e.recoverable, stack: e.stack };
        return [{ isError: true, value: { appError: w } }, []];
      }
      return prev.serialize(v);
    },
    deserialize(s: unknown) {
      const w = (s as { value?: { appError?: Wire } }).value?.appError;
      if (w) {
        const e = new AppError(w.code, w.message, w.detail, w.recoverable);
        e.stack = w.stack;
        throw e;
      }
      return prev.deserialize(s);
    },
  } as ThrowHandler);
}
