"""Build registry.json (ARCHITECTURE 17.5) for weights/dist and optionally upload the tree to Hugging Face.

Usage:
  python tools/publish.py                         # writes weights/dist/registry.json (revision "local")
  python tools/publish.py --upload <user>/web-rvc-models [--public]
      1) uploads every model file, 2) writes registry.json with that commit as revision, 3) uploads it.
      Prints the VITE_MODEL_BASE_URL to use. Needs `hf auth login` (or HF_TOKEN) beforehand.
  --app-registry   also copy registry.json to src/models/registry.gen.json (the app build input)
"""
import argparse
import hashlib
import json
import os
import shutil

from common import DIST, ROOT

PUBLIC = {  # model id -> (folder, basename)
    "contentvec": ("contentvec", "contentvec"),
    "rmvpe": ("rmvpe", "rmvpe"),
    "uvr-hp-vocal": ("uvr", "uvr-hp-vocal"),
}
REGISTRY = "registry.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_files():
    out = []
    for d, _, names in os.walk(DIST):
        for n in names:
            rel = os.path.relpath(os.path.join(d, n), DIST).replace(os.sep, "/")
            if rel != REGISTRY:
                out.append(rel)
    return sorted(out)


def build_registry(revision):
    files = tree_files()
    reg = {"format": 1, "revision": revision,
           "files": {f: {"bytes": os.path.getsize(os.path.join(DIST, f)), "sha256": sha256(os.path.join(DIST, f))}
                     for f in files},
           "models": {}, "rvcTemplates": {}}
    for mid, (folder, base) in PUBLIC.items():
        entry = {k: f"{folder}/{base}.{ext}" for k, ext in (("graph", "onnx"), ("weights", "pth"), ("manifest", "manifest.json"))}
        if mid == "uvr-hp-vocal":
            entry["params"] = json.load(open(os.path.join(DIST, entry["manifest"])))["meta"]["params"]
        reg["models"][mid] = entry
    for f in files:
        if f.startswith("rvc/") and f.endswith(".manifest.json"):
            man = json.load(open(os.path.join(DIST, f)))
            reg["rvcTemplates"][man["meta"]["signature"]] = {"id": man["id"], "graph": f"rvc/{man['id']}.onnx", "manifest": f}
    missing = [p for e in reg["models"].values() for k, p in e.items() if p not in reg["files"]]
    missing += [p for e in reg["rvcTemplates"].values() for p in (e["graph"], e["manifest"]) if p not in reg["files"]]
    assert not missing, f"run the export scripts first, missing: {missing}"
    assert len(reg["rvcTemplates"]) == 12, f"expected 12 RVC templates, got {len(reg['rvcTemplates'])}"
    return reg


def write_registry(reg):
    path = os.path.join(DIST, REGISTRY)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=1)
    total = sum(v["bytes"] for v in reg["files"].values())
    print(f"{path}: {len(reg['files'])} files, {total / 1e6:.1f} MB, revision {reg['revision']}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", metavar="REPO_ID", help="Hugging Face model repo, e.g. me/web-rvc-models")
    ap.add_argument("--public", action="store_true", help="create the repo public (default private, R11)")
    ap.add_argument("--app-registry", action="store_true", help="also write src/models/registry.gen.json")
    a = ap.parse_args()

    shutil.copyfile(os.path.join(os.path.dirname(__file__), "LICENSES.md"), os.path.join(DIST, "LICENSES.md"))
    revision = "local"
    if a.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.upload, repo_type="model", private=not a.public, exist_ok=True)
        info = api.upload_folder(folder_path=DIST, repo_id=a.upload, ignore_patterns=[REGISTRY],
                                 commit_message="web-rvc model files")
        revision = info.oid
    path = write_registry(build_registry(revision))
    if a.upload:
        info = api.upload_file(path_or_fileobj=path, path_in_repo=REGISTRY, repo_id=a.upload,
                               commit_message=f"registry for {revision[:12]}")
        print(f"VITE_MODEL_BASE_URL=https://huggingface.co/{a.upload}/resolve/{info.oid}/")
    if a.app_registry:
        dst = os.path.join(ROOT, "src", "models", "registry.gen.json")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(path, dst)
        print("copied to", dst)


if __name__ == "__main__":
    main()
