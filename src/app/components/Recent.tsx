// DESIGN 9.4: recent work above the input cards (I-2). Hidden when nothing is stored.
import { History, Trash2 } from 'lucide-react';
import { fmtSigned, fmtWhen } from '../lib/format';
import { cn } from '../lib/utils';
import { openRecent, openSettings, removeRecent } from '../controller';
import { selBusy, useApp } from '../store';
import { useT } from '../i18n';

export function RecentList() {
  const t = useT();
  const list = useApp((s) => s.recent);
  const busy = useApp(selBusy);
  if (!list?.length) return null;
  return (
    <section className={cn('mb-5 transition-opacity', busy && 'opacity-55')} inert={busy} aria-labelledby="recent-h">
      <div className="flex items-baseline gap-3 mb-2">
        <h3 id="recent-h" className="text-h3">{t('recent.title')}</h3>
        <span className="text-caption text-muted-foreground">{t('recent.sub')}</span>
        <button type="button" className="ml-auto text-small font-semibold text-accent-foreground hover:underline" onClick={() => openSettings('storage')}>{t('recent.manage')}</button>
      </div>
      <ul className="grid gap-3 grid-cols-1 md:grid-cols-3">
        {list.map((r) => {
          // The newest job of a song with an older sibling gets "A/B" (list is newest first).
          const older = list.find((o) => o !== r && o.songName === r.songName && o.createdAt < r.createdAt);
          const newest = !list.some((o) => o.songName === r.songName && o.createdAt > r.createdAt);
          return (
            <li key={r.id} className="group relative flex items-center gap-3 h-16 rounded-xl bg-card shadow-card pl-3 pr-2">
              <span className="grid place-items-center size-9 rounded-full bg-accent text-accent-foreground shrink-0"><History className="size-4" /></span>
              <button type="button" className="flex-1 min-w-0 text-left after:absolute after:inset-0 after:rounded-xl focus-visible:outline-none focus-visible:after:ring-2 focus-visible:after:ring-ring"
                onClick={() => void openRecent(r.id)}>
                <span className="block font-semibold truncate">{r.songName}</span>
                <span className="block text-caption text-muted-foreground truncate tnum">{t('recent.line', { model: r.modelName, k: fmtSigned(r.f0UpKey), when: fmtWhen(t, r.createdAt) })}</span>
              </button>
              {newest && older && (
                <button type="button" className="relative z-10 shrink-0 h-8 px-2 rounded-lg text-caption font-semibold text-accent-foreground hover:bg-muted" onClick={() => void openRecent(r.id, older.id)}>{t('recent.ab')}</button>
              )}
              <button type="button" className="relative z-10 grid place-items-center size-8 shrink-0 rounded-lg text-muted-foreground hover:bg-muted hover:text-destructive"
                aria-label={t('recent.remove', { name: r.songName })} title={t('common.delete')} onClick={() => void removeRecent(r.id)}><Trash2 className="size-4" /></button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
