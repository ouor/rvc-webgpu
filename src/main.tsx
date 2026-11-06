// App shell: cross-origin isolation (production), global styles, React root.
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { ensureIsolation } from './coi';
import { App } from './app/App';
import './app/styles/globals.css';

void ensureIsolation();
createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
