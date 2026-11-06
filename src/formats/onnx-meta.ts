// Minimal ModelProto reader (ARCHITECTURE §7.4): metadata_props and graph inputs, without full protobuf decoding.
// onnxruntime-web does not expose model metadata; Applio stores sr/version/f0 in metadata "model_info".

type Field = { f: number; wt: number; v: number; s: number; e: number };

function varint(b: Uint8Array, p: number): [number, number] {
  let v = 0, mul = 1;
  for (;;) {
    if (p >= b.length) throw new Error('truncated varint');
    const x = b[p++];
    v += (x & 0x7f) * mul;
    if (!(x & 0x80)) return [v, p];
    mul *= 128;
  }
}

function* fields(b: Uint8Array, start = 0, end = b.length): Generator<Field> {
  let p = start;
  while (p < end) {
    let key: number;
    [key, p] = varint(b, p);
    const f = Math.floor(key / 8), wt = key & 7;
    if (wt === 0) { let v: number; [v, p] = varint(b, p); yield { f, wt, v, s: 0, e: 0 }; }
    else if (wt === 1) p += 8;
    else if (wt === 5) p += 4;
    else if (wt === 2) { let n: number; [n, p] = varint(b, p); yield { f, wt, v: 0, s: p, e: p + n }; p += n; }
    else throw new Error('unsupported wire type ' + wt);
  }
}

const sub = (b: Uint8Array, x: Field, f: number) => [...fields(b, x.s, x.e)].filter((y) => y.f === f);
const td = new TextDecoder();
const text = (b: Uint8Array, x: Field | undefined) => (x ? td.decode(b.subarray(x.s, x.e)) : '');

export interface OnnxInput { name: string; elemType: number; dims: (number | string)[] } // elemType: 1 f32, 7 i64, 10 f16

export interface OnnxInfo { metadata: Record<string, string>; inputs: OnnxInput[]; outputs: OnnxInput[] }

/** Throws on malformed protobuf; callers map that to ONNX_UNRECOGNIZED. */
export function readOnnxInfo(model: Uint8Array): OnnxInfo {
  const metadata: Record<string, string> = {};
  const inputs: OnnxInput[] = [], outputs: OnnxInput[] = [];
  const valueInfo = (vi: Field): OnnxInput => {
    const tensor = sub(model, vi, 2).flatMap((t) => sub(model, t, 1))[0];
    const elem = tensor ? sub(model, tensor, 1)[0] : undefined;
    const dims = tensor ? sub(model, tensor, 2).flatMap((sh) => sub(model, sh, 1)).map((d) => {
      const [x] = [...fields(model, d.s, d.e)];
      return x?.f === 1 ? x.v : x?.f === 2 ? text(model, x) : '?';
    }) : [];
    return { name: text(model, sub(model, vi, 1)[0]), elemType: elem?.v ?? 0, dims };
  };
  for (const top of fields(model)) {
    if (top.f === 14 && top.wt === 2) metadata[text(model, sub(model, top, 1)[0])] = text(model, sub(model, top, 2)[0]);
    if (top.f === 7 && top.wt === 2) {
      const inits = new Set(sub(model, top, 5).map((i) => text(model, sub(model, i, 8)[0])));
      for (const vi of sub(model, top, 11)) { const x = valueInfo(vi); if (!inits.has(x.name)) inputs.push(x); }
      for (const vi of sub(model, top, 12)) outputs.push(valueInfo(vi));
    }
  }
  return { metadata, inputs, outputs };
}
