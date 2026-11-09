import hashlib
import json
import sys
from pathlib import Path

root, prepared, ref, day = map(Path, sys.argv[1:])
expected = json.loads(day.read_bytes())
manifest = (prepared / "manifest.json").read_bytes()
if hashlib.sha256(manifest).hexdigest() != expected["source_manifest_sha256"]:
    raise RuntimeError("prepared source manifest SHA mismatch")
records = json.loads(manifest)["files"]
identity = json.loads((ref / "source-identity.json").read_bytes())
previous = 0
for row in expected["files"]:
    index = row["index"]
    spec = records[index]
    if any(row[k] != spec[k] for k in row) or previous >= row["min_timestamp_ns"]:
        raise RuntimeError("day source identity/time overlap")
    previous = row["max_timestamp_ns"]
    for name, path in [("raw", Path(spec["source"])), ("prepared", prepared / f"{index:03}.sorted.tsv")]:
        saved = identity[name][index]
        st = path.stat()
        if (str(path), st.st_size, st.st_mtime_ns) != (saved["path"], saved["bytes"], saved["mtime_ns"]):
            raise RuntimeError(f"changed {path}")
        if name == "raw" and saved["sha256"] != row["source_sha256"]:
            raise RuntimeError(f"raw digest differs {path}")
print("Verified manifest SHA, 24 disjoint source ranges, recorded raw SHA and raw/sorted stat identity", flush=True)
