"""Encode one cold segment directly from its starting line in the sorted hourly TSVs."""
import bisect
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path("/work")
segment = int(sys.argv[1])
assert 2 <= segment <= 6
start = (segment - 1) * 100
end = segment * 100
catalog = json.loads((root/"prep/day-catalog.json").read_text())["chunks"]
hours = json.loads((root/"sorted/manifest.json").read_text())["files"]
offsets = [0, *itertools.accumulate(h["counts"]["valid_records"] for h in hours)]
hour = bisect.bisect_right(offsets, start * 100000) - 1
last = bisect.bisect_left(offsets, end * 100000)
files = [str(root/"sorted"/f"{h['index']:03}.sorted.tsv") for h in hours[hour:last]]
skip = start * 100000 - offsets[hour]
scratch = root/f"range-{segment}"
scratch.mkdir(exist_ok=True)
dest = root/f"segment-{segment}"
dest.mkdir(exist_ok=True)
feed = subprocess.Popen(["bash", "-c", 'tail -n "+$1" "$2" || exit $?; shift 2; if [ "$#" -gt 0 ]; then cat "$@"; fi',
                         "range", str(skip+1), *files], stdout=subprocess.PIPE)
pack = subprocess.Popen([root/"prep/pack_range", scratch, str(start), str(end)], stdin=feed.stdout)
feed.stdout.close()
try:
    for i in range(start, end):
        row = catalog[i]
        path = scratch/row["file"]
        while not path.exists():
            if pack.poll() is not None and not path.exists():
                raise RuntimeError(f"packer stopped before {path}")
            time.sleep(0.5)
        with path.open("rb") as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(8<<20),b""):
                digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            raise RuntimeError(f"source ordinal or content mismatch at {i}")
        common = root/"chunks"/row["file"]
        try:
            os.link(path, common)
        except FileExistsError:
            if hashlib.sha256(common.read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError(f"conflicting common chunk {i}")
        path.unlink()
        link = dest/f"{i-start:06}.btcbin"
        try:
            link.symlink_to(common)
        except FileExistsError:
            if link.resolve()!=common: raise
        if i==start:
            manifest=dict(first_request=start*100000+1,last_request=end*100000,requests=10000000,
                          input_tokens=sum(r["input_tokens"] for r in catalog[start:end]),initial_cache="empty")
            tmp=dest/"range-start.tmp"
            tmp.write_text(json.dumps(manifest))
            tmp.replace(dest/"segment.json")
            print("verified first chunk",i,flush=True)
    if pack.wait()!=0: raise RuntimeError("pack failed")
finally:
    if pack.poll() is None: pack.terminate()
    if feed.poll() is None: feed.terminate()
