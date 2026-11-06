// Display role of a public model / download item. Names come from the engine (DownloadItem.label, CacheEntry.name).
import { PUBLIC_IDS, type PublicId } from '@/shared/types';

const ROLE: Record<PublicId, 'pm.separate' | 'pm.f0' | 'pm.features'> = {
  [PUBLIC_IDS.separator]: 'pm.separate', [PUBLIC_IDS.rmvpe]: 'pm.f0', [PUBLIC_IDS.contentvec]: 'pm.features',
};
export type Role = (typeof ROLE)[PublicId] | 'pm.template';
/** Any other public id is a conversion template (ENGINE_API: templates are the remaining public ids). */
export const roleOf = (id: string): Role => ROLE[id as PublicId] ?? 'pm.template';
