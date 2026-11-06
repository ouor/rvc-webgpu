import { CircleCheckIcon, InfoIcon, Loader2Icon, OctagonXIcon, TriangleAlertIcon } from 'lucide-react';
import { Toaster as Sonner, type ToasterProps } from 'sonner';

// DESIGN 9.8: top-center under the header (clear of the bottom bar and progress panel), popover surface in both themes,
// only the icon carries the status colour; 4 s, 6 s with an action, at most 2 visible.
const Toaster = (props: ToasterProps) => (
  <Sonner
    className="toaster group"
    position="top-center"
    offset={72}
    visibleToasts={2}
    duration={4000}
    icons={{
      success: <CircleCheckIcon className="size-4" />,
      info: <InfoIcon className="size-4" />,
      warning: <TriangleAlertIcon className="size-4" />,
      error: <OctagonXIcon className="size-4" />,
      loading: <Loader2Icon className="size-4 animate-spin" />,
    }}
    toastOptions={{ classNames: {
      toast: '!bg-popover !text-popover-foreground !border !border-border !rounded-xl !shadow-pop !max-w-[420px]',
      description: '!text-muted-foreground', actionButton: '!bg-primary !text-primary-foreground',
      success: '[&_[data-icon]]:text-success', error: '[&_[data-icon]]:text-destructive', warning: '[&_[data-icon]]:text-warning', info: '[&_[data-icon]]:text-accent-foreground',
    } }}
    {...props}
  />
);

export { Toaster };
