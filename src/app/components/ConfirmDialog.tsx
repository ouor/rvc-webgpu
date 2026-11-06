// Global confirm dialog (store.confirm, opened by confirmAsync in actions).
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from './ui/alert-dialog';
import { answerConfirm } from '../controller';
import { useApp } from '../store';
import { useT } from '../i18n';

export function ConfirmDialog() {
  const t = useT();
  const c = useApp((s) => s.confirm);
  const done = answerConfirm;
  return (
    <AlertDialog open={!!c} onOpenChange={(o) => !o && done(false)}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="text-h2">{c?.title}</AlertDialogTitle>
          <AlertDialogDescription className="text-body">{c?.body}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => done(false)}>{t('common.cancel')}</AlertDialogCancel>
          {c?.alt && <AlertDialogAction variant="secondary" onClick={() => done('alt')}>{c.alt}</AlertDialogAction>}
          <AlertDialogAction variant={c?.destructive ? 'destructive' : 'default'} onClick={() => done(true)}>{c?.action}</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
