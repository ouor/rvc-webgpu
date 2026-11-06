// Display role of a public model / cache item. Names come from the engine (DownloadItem.label, CacheEntry.name).
import { DEFAULT_SEPARATE } from '@/shared/types';

// shortcut: the engine exports only the separator id; 'rmvpe'/'contentvec' are matched by string until DownloadItem/CacheEntry carry a kind (core request).
const ROLE = { [DEFAULT_SEPARATE.modelId]: 'pm.separate', rmvpe: 'pm.f0', contentvec: 'pm.features' } as const;
export type Role = (typeof ROLE)[keyof typeof ROLE] | 'pm.template';
export const roleOf = (id: string): Role => (ROLE as Record<string, Role>)[id] ?? 'pm.template';
