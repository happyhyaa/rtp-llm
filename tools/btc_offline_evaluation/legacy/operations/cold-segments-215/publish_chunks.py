"""Publish each verified segment chunk as soon as it is available to the blocking reader."""
import hashlib
import json
import time
from pathlib import Path

root = Path("/work")
catalog = json.loads((root/"prep/day-catalog.json").read_text())["chunks"]
for segment in range(2, 7):
    dest = root/f"segment-{segment}"
    dest.mkdir(exist_ok=True)
    first = (segment - 1) * 100
    rows = catalog[first:first+100]
    try:
        for local, row in enumerate(rows):
            chunk = root/"chunks"/row["file"]
            while not chunk.is_file():
                time.sleep(1)
            digest = hashlib.sha256()
            with chunk.open("rb") as f:
                for block in iter(lambda: f.read(8 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError(f"SHA mismatch: {chunk}")
            link = dest/f"{local:06}.btcbin"
            try:
                link.symlink_to(chunk)
            except FileExistsError:
                if link.resolve() != chunk:
                    raise
            if local == 0 and not (dest/"segment.json").exists():
                tmp = dest/"first-verified.tmp"
                tmp.write_text(json.dumps(dict(first_request=first*100000+1,
                    last_request=(first+100)*100000, requests=10000000,
                    input_tokens=sum(r["input_tokens"] for r in rows), initial_cache="empty")))
                tmp.replace(dest/"segment.json")
                print(segment, "first chunk verified", flush=True)
    except Exception as error:
        (dest/"FAILED.json").write_text(json.dumps({"error":str(error)}))
        raise
