"""Shared helpers for the web-rvc model tools (ARCHITECTURE.md 17: tools <-> src contract).

A "template" is a weightless ONNX graph whose parameters live in an external "weights.bin" that the
browser builds from a .pth state_dict and a manifest (17.2). Everything here produces or consumes that.
"""
import io
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import onnx
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_IN = os.path.join(ROOT, "external", "weight")
DIST = os.path.join(ROOT, "weights", "dist")          # same relative layout as the HF/R2 tree
REFERENCE = os.path.join(ROOT, "weights", "reference")
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
sys.path.insert(0, os.path.join(VENDOR, "rvc"))      # infer_pack, rmvpe (rvc-nano)
sys.path.insert(0, VENDOR)                           # uvr

EXTERNAL_DATA = "weights.bin"
OPSET = 17


def export_template(module, args, out_dir, name, *, id, inputs, outputs, meta, pth=True, half=True):
    """Export `module` to <out_dir>/<name>.onnx with every state_dict tensor moved to external data.

    inputs/outputs: lists of IoSpec dicts {name, dtype, shape} (shape strings = dynamic axes).
    pth: also write <name>.pth (flat OrderedDict, fp16 unless half=False) holding exactly the manifest keys.
    Returns the manifest dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    dynamic_axes = {io_["name"]: {i: d for i, d in enumerate(io_["shape"]) if isinstance(d, str)}
                    for io_ in inputs + outputs}
    buf = io.BytesIO()
    torch.onnx.export(module, args, buf, input_names=[i["name"] for i in inputs],
                      output_names=[o["name"] for o in outputs], dynamic_axes=dynamic_axes,
                      opset_version=OPSET, do_constant_folding=False, dynamo=False)
    model = onnx.load_from_string(buf.getvalue())

    random_ops = sorted({n.op_type for n in model.graph.node if n.op_type.startswith("Random")})
    assert not random_ops, f"{id}: graph contains {random_ops}"

    sd = module.state_dict()
    inits = {t.name: t for t in model.graph.initializer}
    tensors, offset, weights = [], 0, OrderedDict()
    for key, value in sd.items():  # state_dict order -> stable, readable layout
        t = inits.get(key)
        if t is None:
            continue  # unused by the graph (num_batches_tracked, aux heads, enc_q, ...)
        assert list(t.dims) == list(value.shape) and t.data_type == onnx.TensorProto.FLOAT, key
        n = value.numel() * 4
        t.ClearField("raw_data")
        t.ClearField("float_data")
        t.data_location = onnx.TensorProto.EXTERNAL
        del t.external_data[:]
        for k, v in (("location", EXTERNAL_DATA), ("offset", str(offset)), ("length", str(n))):
            e = t.external_data.add()
            e.key, e.value = k, v
        tensors.append({"key": key, "shape": list(value.shape), "offset": offset})
        weights[key] = value.detach().cpu().contiguous()
        offset += n
    inline = [n for n in inits if n not in weights]

    onnx.save(model, os.path.join(out_dir, name + ".onnx"))
    manifest = {"format": 1, "id": id, "externalData": EXTERNAL_DATA, "bytes": offset, "tensors": tensors,
                "inputs": inputs, "outputs": outputs, "meta": meta}
    with open(os.path.join(out_dir, name + ".manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    if pth:
        torch.save(OrderedDict((k, v.half() if half else v.float()) for k, v in weights.items()),
                   os.path.join(out_dir, name + ".pth"))
    print(f"[{id}] graph {os.path.getsize(os.path.join(out_dir, name + '.onnx')) / 1e6:.2f} MB, "
          f"{len(tensors)} external tensors ({offset / 1e6:.1f} MB fp32), inline initializers: {len(inline)}")
    return manifest


def io_spec(name, dtype, shape):
    return {"name": name, "dtype": dtype, "shape": list(shape)}


def load_session(onnx_path, manifest_path, state_dict, providers=("CPUExecutionProvider",)):
    """Python twin of src models/loader.ts: inject state_dict (fp16/fp32) per manifest, return an ORT session."""
    import onnxruntime as ort

    man = json.load(open(manifest_path, encoding="utf-8"))
    model = onnx.load(onnx_path, load_external_data=False)
    by_name = {t.name: t for t in model.graph.initializer}
    for m in man["tensors"]:
        v = state_dict[m["key"]]
        assert list(v.shape) == m["shape"], m["key"]
        t = by_name[m["key"]]
        del t.external_data[:]
        t.data_location = onnx.TensorProto.DEFAULT
        t.raw_data = v.float().contiguous().numpy().astype("<f4").tobytes()
    return ort.InferenceSession(model.SerializeToString(), providers=list(providers))


def rvc_signature(version, f0, cfg):
    """17.4: registry key of an RVC template. Must equal the TS JSON.stringify output."""
    norm = lambda v: [norm(x) for x in v] if isinstance(v, (list, tuple)) else int(v)
    sig = [version, int(f0), *[norm(cfg[i]) for i in range(2, 8)], str(cfg[9]),
           *[norm(cfg[i]) for i in range(10, 15)], norm(cfg[16]), norm(cfg[17])]
    return json.dumps(sig, separators=(",", ":"))


def write_case(case_dir, source, inputs, outputs, meta):
    """17.6 fixture: <case_dir>/case.json + one little-endian .bin per array."""
    os.makedirs(case_dir, exist_ok=True)
    dtypes = {np.dtype("float32"): "float32", np.dtype("float64"): "float64", np.dtype("int64"): "int64"}

    def dump(arrays):
        spec = {}
        for k, a in arrays.items():
            a = np.ascontiguousarray(a)
            dt = dtypes[a.dtype]  # KeyError = unsupported dtype, convert at the call site
            a.astype(a.dtype.newbyteorder("<")).tofile(os.path.join(case_dir, k + ".bin"))
            spec[k] = {"file": k + ".bin", "dtype": dt, "shape": list(a.shape)}
        return spec

    case = {"format": 1, "source": source, "inputs": dump(inputs), "outputs": dump(outputs), "meta": meta}
    with open(os.path.join(case_dir, "case.json"), "w", encoding="utf-8") as f:
        json.dump(case, f, indent=1)


def snr_db(ref, got):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    return float(10 * np.log10((ref ** 2).sum() / max(((ref - got) ** 2).sum(), 1e-300)))


if __name__ == "__main__":  # self-check of the contract example in 17.4
    kyubey = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3, [10, 10, 2, 2], 512,
              [16, 16, 4, 4], 109, 256, 40000]
    want = '["v2",1,192,192,768,2,6,3,"1",[3,7,11],[[1,3,5],[1,3,5],[1,3,5]],[10,10,2,2],512,[16,16,4,4],256,40000]'
    assert rvc_signature("v2", True, kyubey) == want, rvc_signature("v2", True, kyubey)
    print("ok")
