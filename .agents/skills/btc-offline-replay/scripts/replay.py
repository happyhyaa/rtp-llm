"""Run a caller-supplied native metadata replay executable."""
import argparse
import json
import struct
import subprocess
from pathlib import Path

from prepare import CHUNK, digest


def token_prefix(path, count):
    with path.open("rb") as f:
        magic, base, size, _, name_len = struct.unpack("<8sIQQI", f.read(32))
        if magic != b"BTCPFX01" or count > size:
            raise ValueError("invalid chunk header")
        f.seek(name_len, 1)
        total = 0
        for _ in range(count):
            _, length, rid_len = struct.unpack("<QII", f.read(16))
            total += length
            f.seek(rid_len + length // base * 8, 1)
        return total


def prefix(data, chunks, count):
    total = 0
    for chunk in chunks:
        if count == 0:
            break
        if count < chunk["requests"]:
            return total + token_prefix(data / chunk["file"], count)
        count -= chunk["requests"]
        total += chunk["input_tokens"]
    if count:
        raise ValueError("requested range exceeds manifest")
    return total


def main(a):
    manifest = a.data / "manifest.json"
    trace = json.loads(manifest.read_bytes())
    if trace["format"] != "BTCPFX01":
        raise ValueError("unsupported trace format")
    for i, c in enumerate(trace["chunks"]):
        if c["file"] != f"{i:06}.btcbin" or digest(a.data / c["file"]) != c["sha256"]:
            raise ValueError(f"chunk {i} failed identity check")
        if i + 1 < len(trace["chunks"]) and c["requests"] != CHUNK:
            raise ValueError("non-final chunk must contain 100000 records")
    if a.warmup < 0 or a.measure < 1 or a.stop is not None and not 0 < a.stop <= a.measure:
        raise ValueError("invalid replay range")
    if a.warmup + a.measure > trace["requests"]:
        raise ValueError("insufficient requests in the trace")
    end = a.stop or a.measure
    expected = prefix(a.data, trace["chunks"], a.warmup + end) - prefix(a.data, trace["chunks"], a.warmup)
    binary = a.backend.resolve(strict=True)
    config = a.config.read_bytes()
    identity = dict(trace_sha256=digest(manifest), backend_sha256=digest(binary),
                    config_sha256=digest(a.config), revision=a.revision, warmup=a.warmup, measure=a.measure)
    previous = 0
    if a.resume:
        old = json.loads((a.resume / "identity.json").read_bytes())
        if old != identity or config != (a.resume / "parameters.json").read_bytes() or not (a.resume / "SUCCESS").exists():
            raise ValueError("resume requires a verified run with identical trace, binary, configuration and interval")
        previous = json.loads((a.resume / "native/summary.json").read_bytes())["total"]["requests"]
        if previous >= end:
            raise ValueError("resume endpoint must advance")
    a.out.mkdir(parents=True, exist_ok=False)
    ident = a.out / "identity.json"
    ident.write_text(json.dumps(identity, sort_keys=True) + "\n")
    cfg = a.out / "parameters.json"
    cfg.write_bytes(config)
    native = a.out / "native"
    cmd = [str(binary), str(cfg), str(a.data), str(native), "--identity", str(ident),
           "--requests", str(a.warmup + a.measure), "--warmup", str(a.warmup),
           "--stop-after", str(a.warmup + end)]
    if a.resume:
        cmd.extend(["--resume", str(a.resume / "native/checkpoint")])
    subprocess.run(cmd, check=True)
    t = json.loads((native / "summary.json").read_bytes())["total"]
    if (t["requests"], t["input_tokens"]) != (end, expected):
        raise ValueError("backend request or input-token denominator mismatch")
    if not 0 <= t["joint_tokens"]["sum"] <= t["full_tokens"]["sum"] <= expected:
        raise ValueError("invalid aggregate match lengths")
    if (native / "match-lengths.u64").stat().st_size != 16 * (end - previous):
        raise ValueError("unexpected number of per-request match pairs")
    (a.out / "SUCCESS").touch()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("backend", "data", "config", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--measure", type=int, required=True)
    p.add_argument("--stop", type=int, help="cumulative measured count for this stage")
    p.add_argument("--resume", type=Path, help="previous verified run directory")
    main(p.parse_args())
