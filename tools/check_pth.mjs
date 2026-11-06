// 17.3 check: each public .pth parses with the browser allowlist reader and matches its manifest.
// Usage: node tools/check_pth.mjs <pth-reader.mjs> <file.pth>...
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [reader, ...files] = process.argv.slice(2);
const { readPth } = await import(pathToFileURL(reader).href);
let failed = 0;
for (const f of files) {
  const { root } = readPth(new Uint8Array(readFileSync(f)));
  const man = JSON.parse(readFileSync(f.replace(/\.pth$/, '.manifest.json'), 'utf8'));
  const bad = man.tensors.filter((t) => root.get(t.key)?.shape.join() !== t.shape.join());
  console.log(`${f}: ${root.size} tensors, manifest ${man.tensors.length}, mismatches ${bad.length}`);
  failed += bad.length;
}
process.exit(failed ? 1 : 0);
