"""Verify the source hashes and use independently recorded qps_worker hits."""
import hashlib
import json
import struct
from pathlib import Path

base = Path("/data0/REPLAY_USER/hit-rate-evaluation")
grid = base / "cp4-gridsearch-20261103-v1"
group = grid / "finite/step-01-n-00"
summary = json.loads((group / "summary.json").read_text())
assert [summary[k] for k in ("full_capacity_blocks", "swa_capacity_blocks", "last_n", "linear_step")] == [3218236, 3218236, 0, 1]
requests = 0
input_tokens = 0
last = 0
first = None
for index, entry in enumerate(json.loads((grid / "input-identity.json").read_text())[:100]):
    chunk = Path(entry["path"])
    assert chunk.name == f"{index:06d}.bin"
    data = chunk.read_bytes()
    assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
    magic, capacity, count = struct.unpack_from("<8sQQ", data)
    assert magic == b"FNLAST01" and count == 100000
    start = 24
    for _ in range(count):
        ns, tokens, blocks, _ = struct.unpack_from("<QQII", data, start)
        assert ns >= last and blocks == tokens // 1024
        last = ns
        if first is None:
            first = ns
        start += 24 + 8 * blocks
        input_tokens += tokens
    assert start == len(data)
    requests += count
full = joint = 0
digest = hashlib.sha256()
with (group / "hits.u32").open("rb") as stream:
    for _ in range(100):
        buf = stream.read(800000)
        assert len(buf) == 800000
        digest.update(buf)
        for f, h in struct.iter_unpack("<II", buf):
            full += f
            joint += h
print(json.dumps(dict(requests=requests, input_tokens=input_tokens, historical_hits=str(group / "hits.u32"), historical_prefix_sha256=digest.hexdigest(), full_hit_tokens=full*1024, hit_tokens=joint*1024, first_ns=first, last_ns=last, source_chunks_verified=100), indent=2))
