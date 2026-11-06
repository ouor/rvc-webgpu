// Minimal shell. Product UI (screens S1-S7) is built by the UI developer on top of app/engine-client.ts.
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { ensureIsolation } from './coi';
import { App } from './app/App';

void ensureIsolation();
createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
