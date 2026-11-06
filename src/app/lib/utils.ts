import { clsx, type ClassValue } from 'clsx';
import { extendTailwindMerge } from 'tailwind-merge';

// Design-token font sizes (text-body, text-lead...) must not be mistaken for text colours when merging.
const twMerge = extendTailwindMerge({ extend: { classGroups: { 'font-size': [{ text: ['caption', 'small', 'body', 'lead', 'h3', 'h2', 'h1', 'display'] }] } } });

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));
