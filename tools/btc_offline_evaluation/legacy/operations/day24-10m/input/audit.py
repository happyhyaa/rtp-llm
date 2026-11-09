"""Verify the raw and sorted source bytes for the selected cohort and hash published chunks."""
import concurrent.futures
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
REFERENCE = BASE / "sgp04-cp4-b256-20261102-v3/input1024"
cohort = json.loads((ROOT / "cohort.json").read_bytes())
end_file = cohort["last"]["file_index"]
identity = json.loads((REFERENCE / "source-identity.json").read_bytes())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_source(saved):
    path = Path(saved["path"])
    actual = digest(path)
    st = path.stat()
    if (actual, st.st_size, st.st_mtime_ns) != (saved["sha256"], saved["bytes"], saved["mtime_ns"]):
        raise ValueError(f"source identity changed: {path}")
    return saved


def verify_chunk(path):
    with path.open("rb") as f:
        magic, base, count, tokens, length = struct.unpack("<8sIQQI", f.read(32))
        if (magic, base, count, length, f.read(length)) != (b"BTCPFX01", 256, 100000, 6, b"global"):
            raise ValueError(f"bad header: {path}")
    return dict(file=path.name, requests=count, input_tokens=tokens, bytes=path.stat().st_size, sha256=digest(path))


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    sources = list(pool.map(verify_source, identity["raw"][:end_file+1] + identity["prepared"][:end_file+1]))
    chunks = list(pool.map(verify_chunk, sorted(ROOT.glob("*.btcbin"))))
if len(chunks) != 100 or sum(c["input_tokens"] for c in chunks) != cohort["input_tokens"]:
    raise ValueError("chunk count/token sum mismatch")
audit = dict(cohort=cohort, sources=sources, chunks=chunks)
temp = ROOT / "verification.json.tmp"
temp.write_text(json.dumps(audit, indent=2) + "\n")
temp.replace(ROOT / "verification.json")
