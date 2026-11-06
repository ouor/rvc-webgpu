import { CircleCheckIcon, InfoIcon, Loader2Icon, OctagonXIcon, TriangleAlertIcon } from 'lucide-react';
import { Toaster as Sonner, type ToasterProps } from 'sonner';

// DESIGN Toast: bottom-center, 4 s, dark surface. Theme comes from the app's own setting (no next-themes).
const Toaster = (props: ToasterProps) => (
  <Sonner
    className="toaster group"
    position="bottom-center"
    duration={4000}
    icons={{
      success: <CircleCheckIcon className="size-4" />,
      info: <InfoIcon className="size-4" />,
      warning: <TriangleAlertIcon className="size-4" />,
      error: <OctagonXIcon className="size-4" />,
      loading: <Loader2Icon className="size-4 animate-spin" />,
    }}
    toastOptions={{ classNames: { toast: '!bg-foreground !text-background !border-0 !rounded-xl !shadow-pop', description: '!text-background/70', actionButton: '!bg-background/15 !text-background' } }}
    {...props}
  />
);

export { Toaster };
