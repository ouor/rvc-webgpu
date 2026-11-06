// DESIGN Dropzone: click/Enter/Space opens the picker; dropped files are routed by extension (addFiles).
import type { ReactNode } from 'react';
import { useDropzone } from 'react-dropzone';
import { Bar } from './Bar';
import { cn } from '../lib/utils';
import { addFiles } from '../controller';
import { useT } from '../i18n';

export function Dropzone({ icon, title, hint, accept, multiple, height, parsing, disabled }: {
  icon: ReactNode; title: string; hint: string; accept: string; multiple?: boolean; height: number; parsing?: string; disabled?: boolean;
}) {
  const t = useT();
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (files) => void addFiles(files),
    multiple, disabled: disabled || !!parsing, noDragEventsBubbling: true,
    // react-dropzone's accept map needs MIME types; .pth/.index have none, so the picker filter goes on the input directly.
  });
  if (parsing) {
    return (
      <div className="w-full rounded-xl border-2 border-dashed border-primary/50 bg-accent/40 flex flex-col items-center justify-center gap-3 px-6 text-center" style={{ height }} role="status" aria-live="polite">
        <span className="grid place-items-center size-12 rounded-2xl bg-accent text-accent-foreground">{icon}</span>
        <span className="font-semibold">{parsing}</span>
        <Bar className="w-40" />
      </div>
    );
  }
  return (
    <div
      {...getRootProps({ role: 'button', 'aria-label': `${title}. ${hint}` })}
      className={cn('w-full rounded-xl border-2 border-dashed border-input transition flex flex-col items-center justify-center gap-3 text-center px-4 outline-none',
        'hover:border-primary hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-ring cursor-pointer',
        isDragActive && 'border-solid border-primary bg-accent/60', disabled && 'opacity-55 pointer-events-none')}
      style={{ height }}
    >
      <input {...getInputProps({ accept })} />
      <span className="grid place-items-center size-12 rounded-2xl bg-accent text-accent-foreground">{icon}</span>
      <span className="font-semibold">{isDragActive ? t('drop.here') : title}</span>
      <span className="text-small text-muted-foreground">{hint}</span>
    </div>
  );
}

/** Invisible picker used by "바꾸기" / ".index 넣기" buttons. */
export function pickFiles(accept: string, multiple = false) {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = accept;
  input.multiple = multiple;
  input.onchange = () => input.files && void addFiles([...input.files]);
  input.click();
}
