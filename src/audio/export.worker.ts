// Export worker (ARCHITECTURE 18.3): WAV / MP3 encoding and STORE zips off the main thread and off the engine.
import * as Comlink from 'comlink';
import { installErrorTransfer } from '../shared/rpc';
import { exportApi } from './export-encode';

installErrorTransfer();
Comlink.expose(exportApi);
