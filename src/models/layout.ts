// OPFS layout (ARCHITECTURE §8.2), the only place that builds storage paths.
//   public/<revision>/<registry path>          public models
//   my-models/<id>/{model.pth|model.onnx, model.index, info.json}   saved user models (id = file hash prefix)

export const PUBLIC_ROOT = 'public';
export const MY_MODELS_ROOT = 'my-models';

export const publicDir = (revision: string) => `${PUBLIC_ROOT}/${revision}/`;
export const publicPath = (revision: string, path: string) => publicDir(revision) + path;
/** Revision folder of a stored public path, or undefined. */
export const revisionOf = (storedPath: string) => storedPath.split('/')[1];

export const isMyModelId = (id: string) => /^[0-9a-f]{16}$/.test(id);
export const myModelDir = (id: string) => `${MY_MODELS_ROOT}/${id}`;
export const myModelFile = (id: string, kind: 'pth' | 'onnx' | 'index' | 'info') =>
  `${myModelDir(id)}/${kind === 'info' ? 'info.json' : `model.${kind}`}`;
/** my-model id of a stored path under my-models/, or undefined. */
export const myModelIdOf = (storedPath: string) => (storedPath.startsWith(MY_MODELS_ROOT + '/') ? storedPath.split('/')[1] : undefined);

//   recent/<id>/{job.json, converted.wav, vocals.wav, inst.wav}   recent jobs (I-2); job.json is written last
export const RECENT_ROOT = 'recent';
export const isRecentId = (id: string) => /^[0-9a-z]{6,12}-[0-9a-z]{4}$/.test(id);
export const recentDir = (id: string) => `${RECENT_ROOT}/${id}`;
export const recentFile = (id: string, kind: 'job' | 'converted' | 'vocals' | 'inst') => `${recentDir(id)}/${kind === 'job' ? 'job.json' : `${kind}.wav`}`;
