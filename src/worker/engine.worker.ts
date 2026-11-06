// Engine worker entry: Comlink.expose(createEngine()) (ARCHITECTURE §5).
import * as Comlink from 'comlink';
import { createEngine } from '../engine/engine';
import { installErrorTransfer } from '../shared/rpc';

installErrorTransfer();
Comlink.expose(createEngine());
