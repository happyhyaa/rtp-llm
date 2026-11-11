"""Stream one ordered cache domain into BTCPFX01 chunks."""
import argparse
import gzip
import hashlib
import json
import os
import re
import struct
import subprocess
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

CHUNK = 100000
MASK = (1 << 64) - 1


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(4 << 20), b""):
            h.update(data)
    return h.hexdigest()


@contextmanager
def lines(path):
    if path.suffix == ".zst":
        with subprocess.Popen(["zstd", "-dc", "--", str(path)], stdout=subprocess.PIPE) as proc:
            try:
                yield proc.stdout
            finally:
                proc.stdout.close()
            if proc.wait():
                raise ValueError(f"zstd failed for {path}")
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            yield stream


def bailian(row, base, mode):
    length = row["input_length"]
    hashes = row["input_block_hash_ids"]
    if type(length) is not int or not isinstance(hashes, list) or len(hashes) not in {
        length // base, (length + base - 1) // base
    }:
        raise ValueError("inconsistent input length and hash count")
    keys = []
    prefix = 0
    for value in hashes[:length // base]:
        if not isinstance(value, str) or not re.fullmatch("[0-9a-fA-F]{16}", value):
            raise ValueError("expected 16 hexadecimal digits per hash")
        key = int(value, 16)
        if mode == "raw":
            prefix ^= (key + 0x9e3779b97f4a7c15 + (prefix << 12) + (prefix >> 32)) & MASK
            key = prefix
        keys.append(key if key < 1 << 63 else key - (1 << 64))
    ts = Decimal(str(row["timestamp"])) * 1000000000
    if ts != ts.to_integral_value():
        raise ValueError("timestamp finer than nanoseconds")
    return int(ts), length, keys


def prepare(args, dest):
    domain = args.domain.encode()
    if not 0 < len(domain) <= 1048576 or not 0 < args.base_tokens < 2**31:
        raise ValueError("invalid domain or block size")
    chunks = []
    previous = count = total = 0
    stream = None
    try:
        for path in args.inputs:
            with lines(path) as source:
                for number, line in enumerate(source, 1):
                    try:
                        row = json.loads(line, parse_float=Decimal)
                        if args.format == "bailian":
                            ts, length, keys = bailian(row, args.base_tokens, args.hash_mode)
                        else:
                            if (row["key_form"], row["instance_id"], row["block_tokens"]) != (
                                "prefix", args.domain, args.base_tokens
                            ):
                                raise ValueError("domain, granularity or key form mismatch")
                            ts, length, keys = row["timestamp_ns"], row["input_len"], row["keys"]
                        rid = row["request_id"].encode()
                        if not (type(ts) is int and previous <= ts < 2**64 and ts > 0
                                and type(length) is int and 0 < length < 2**31
                                and isinstance(keys, list) and len(keys) == length // args.base_tokens
                                and len(rid) <= 1048576 and all(type(k) is int for k in keys)):
                            raise ValueError("invalid record or timestamp order")
                        record = struct.pack("<QII", ts, length, len(rid)) + rid + struct.pack(f"<{len(keys)}q", *keys)
                    except (KeyError, ArithmeticError, ValueError, TypeError, AttributeError, struct.error) as e:
                        raise ValueError(f"{path}:{number}: {e}") from e
                    if stream is None:
                        chunk = dest / f"{len(chunks):06}.btcbin"
                        stream = chunk.open("w+b")
                        stream.write(struct.pack("<8sIQQI", b"BTCPFX01", args.base_tokens, 0, 0, len(domain)) + domain)
                        n = tokens = 0
                    stream.write(record)
                    n += 1
                    tokens += length
                    total += length
                    count += 1
                    previous = ts
                    if n == CHUNK:
                        finish(stream, chunk, n, tokens, chunks)
                        stream = None
        if not count:
            raise ValueError("empty trace")
        if stream:
            finish(stream, chunk, n, tokens, chunks)
    finally:
        if stream and not stream.closed:
            stream.close()
    manifest = dict(format="BTCPFX01", base_tokens=args.base_tokens, instance_id=args.domain,
                    hash_mode=args.hash_mode or "prefix", requests=count, input_tokens=total, chunks=chunks)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def finish(stream, path, n, tokens, chunks):
    stream.seek(12)
    stream.write(struct.pack("<QQ", n, tokens))
    stream.close()
    chunks.append(dict(file=path.name, requests=n, input_tokens=tokens, sha256=digest(path)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--format", choices=["bailian", "normalized"], required=True)
    p.add_argument("--hash-mode", choices=["raw", "prefix"])
    p.add_argument("--base-tokens", type=int, required=True)
    p.add_argument("--domain", required=True)
    a = p.parse_args()
    if (a.format == "bailian") != bool(a.hash_mode):
        p.error("specify --hash-mode only for Bailian input; normalized input requires prefix keys")
    if a.out.exists():
        p.error("output already exists")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=a.out.parent) as tmp:
        staged = Path(tmp) / "data"
        staged.mkdir()
        prepare(a, staged)
        os.rename(staged, a.out)
