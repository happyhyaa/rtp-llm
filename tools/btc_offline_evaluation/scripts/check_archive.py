"""Verify the frozen source overlay and its patch without running a replay."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "archive-manifest.json").read_text())
    for name, expected in manifest["source_files"].items():
        data = (root / "legacy/source" / name).read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Source snapshot mismatch: {name}")
    patch = (root / "legacy/source.patch").read_bytes()
    if hashlib.sha256(patch).hexdigest() != manifest["source_patch_sha256"]:
        raise ValueError("Source patch mismatch")
    print(f"Verified {len(manifest['source_files'])} source files and source.patch")


if __name__ == "__main__":
    main()
