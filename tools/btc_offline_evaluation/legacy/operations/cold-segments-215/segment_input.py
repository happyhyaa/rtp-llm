"""Expose one independently verified 10M-request slice as BTCPFX01 chunks 0..99."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path


def prepare(source, catalog, dest, segment):
    rows = json.loads(catalog.read_text())["chunks"]
    first = (segment - 1) * 100
    tokens = 0
    dest.mkdir(parents=True, exist_ok=True)
    for local_index in range(100):
        row = rows[first + local_index]
        chunk = (source / row["file"]).resolve(strict=True)
        if row["file"] != f"{first + local_index:06}.btcbin" or row["requests"] != 100000:
            raise ValueError(f"Invalid ordinal mapping for {chunk}")
        digest = hashlib.sha256()
        with chunk.open("rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError(f"Chunk digest mismatch: {chunk}")
        link = dest / f"{local_index:06}.btcbin"
        if not link.exists():
            link.symlink_to(chunk)
        elif link.resolve() != chunk:
            raise ValueError(f"Conflicting segment input: {link}")
        tokens += row["input_tokens"]
    return {"first_request": first*100000+1, "last_request": (first+100)*100000,
            "requests": 10000000, "input_tokens": tokens, "initial_cache": "empty"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "catalog", "dest"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("segment", type=int, choices=range(1, 7))
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    with (args.dest / ".verify.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = args.dest / "segment.json"
        result = prepare(args.source, args.catalog, args.dest, args.segment)
        tmp = ready.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2)+"\n")
        tmp.replace(ready)
