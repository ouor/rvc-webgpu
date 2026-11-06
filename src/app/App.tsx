import { useEffect } from 'react';
import { Toaster } from './components/ui/sonner';
import { TooltipProvider } from './components/ui/tooltip';
import { AppHeader, Banners, FirstRunNotice, MobileGate } from './components/Chrome';
import { ModelSection, SongCard } from './components/Inputs';
import { OptionsCard } from './components/Options';
import { ActionBar, DownloadDialog, ProgressPanel } from './components/Run';
import { ResultView } from './components/Result';
import { SettingsDialog } from './components/Settings';
import { ConfirmDialog } from './components/ConfirmDialog';
import { addFiles, setOnline, setResolvedTheme } from './controller';
import { cn } from './lib/utils';
import { selBusy, useApp } from './store';

function useTheme() {
  const theme = useApp((s) => s.settings.theme);
  const lang = useApp((s) => s.settings.lang);
  useEffect(() => {
    const mq = matchMedia('(prefers-color-scheme: dark)');
    const apply = () => {
      const dark = theme === 'dark' || (theme === 'system' && mq.matches);
      document.documentElement.classList.toggle('dark', dark);
      setResolvedTheme(dark ? 'dark' : 'light'); // after the class flips, so waveforms read the new colours
    };
    apply();
    mq.addEventListener('change', apply);
    return () => mq.removeEventListener('change', apply);
  }, [theme]);
  useEffect(() => { document.documentElement.lang = lang; }, [lang]);
}

function Work() {
  const busy = useApp(selBusy);
  return (
    <main className="mx-auto max-w-[1280px] px-6 pt-5">
      <FirstRunNotice />
      {/* DESIGN S1: cards are locked (inert, 55 %) while processing so the user still sees what runs. */}
      <div className={cn('grid gap-5 items-start grid-cols-1 lg:grid-cols-2 xl:grid-cols-[1.2fr_1fr_1fr] transition-opacity', busy && 'opacity-55')} inert={busy}>
        <SongCard />
        <ModelSection />
        <div className="lg:col-span-2 xl:col-span-1"><OptionsCard /></div>
      </div>
    </main>
  );
}

export function App() {
  useTheme();
  const view = useApp((s) => s.view);
  const job = useApp((s) => s.job);
  const hasResult = useApp((s) => !!s.result);

  useEffect(() => {
    addEventListener('online', setOnline);
    addEventListener('offline', setOnline);
    // Whole-page drop: files are routed to the right card by extension.
    const over = (e: DragEvent) => e.preventDefault();
    const drop = (e: DragEvent) => { e.preventDefault(); if (e.dataTransfer?.files.length) void addFiles([...e.dataTransfer.files]); };
    addEventListener('dragover', over);
    addEventListener('drop', drop);
    return () => { removeEventListener('online', setOnline); removeEventListener('offline', setOnline); removeEventListener('dragover', over); removeEventListener('drop', drop); };
  }, []);

  return (
    <TooltipProvider delayDuration={300}>
      <div className="min-h-screen pb-40">
        <AppHeader />
        <Banners />
        {view === 'result' && hasResult ? <ResultView /> : <Work />}
        {job ? <ProgressPanel /> : view === 'work' && <ActionBar />}
      </div>
      <DownloadDialog />
      <SettingsDialog />
      <ConfirmDialog />
      <MobileGate />
      <Toaster />
    </TooltipProvider>
  );
}
