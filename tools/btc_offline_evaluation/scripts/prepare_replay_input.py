"""Pack an ordered base-granularity prefix JSONL once for repeated BTC runs."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def pack(source, dest, base):
    if not 0 < base <= 2**31 - 1:
        raise ValueError("invalid base block size")
    dest.mkdir(exist_ok=False)
    pending = dest / "requests.tmp"
    source_digest = hashlib.sha256()
    count = total = previous = 0
    domain = None
    with source.open("rb") as lines, pending.open("wb") as out:
        for ordinal, line in enumerate(lines, 1):
            source_digest.update(line)
            r = json.loads(line)
            ts, length, keys = r["timestamp_ns"], r["input_len"], r["keys"]
            if not (
                r["key_form"] == "prefix"
                and type(r["block_tokens"]) is int
                and r["block_tokens"] == base
                and type(ts) is int
                and 0 < ts < 2**64
                and ts >= previous
                and type(length) is int
                and 0 < length < 2**31
                and isinstance(keys, list)
                and len(keys) == length // base
                and all(type(k) is int and -(2**63) <= k < 2**63 for k in keys)
            ):
                raise ValueError(f"invalid prefix record at line {ordinal}")
            current = r["instance_id"]
            if (
                not isinstance(current, str)
                or not current
                or len(current.encode()) > 1048576
            ):
                raise ValueError("invalid cache domain")
            if domain is None:
                domain = current
                name = domain.encode()
                out.write(
                    struct.pack("<8sIQQI", b"BTCPFX01", base, 0, 0, len(name)) + name
                )
            if current != domain:
                raise ValueError("one cache domain per replay")
            rid = r.get("request_id", str(ordinal))
            if not isinstance(rid, str) or len(rid.encode()) > 1048576:
                raise ValueError("invalid request ID")
            rid = rid.encode()
            out.write(struct.pack("<QII", ts, length, len(rid)) + rid)
            out.write(struct.pack(f"<{len(keys)}q", *keys))
            count, total, previous = ordinal, total + length, ts
        if not count:
            raise ValueError("empty trace")
        out.seek(12)
        out.write(struct.pack("<QQ", count, total))
    checksum = hashlib.sha256()
    with pending.open("rb") as f:
        for part in iter(lambda: f.read(4 << 20), b""):
            checksum.update(part)
    manifest = dict(
        format="BTCPFX01",
        base_tokens=base,
        key_form="prefix",
        instance_id=domain,
        requests=count,
        input_tokens=total,
        external_cp_coarsening=False,
        source_sha256=source_digest.hexdigest(),
        data_sha256=checksum.hexdigest(),
    )
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pending.rename(dest / "requests.btcbin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("normalized_jsonl", type=Path)
    parser.add_argument("new_output_dir", type=Path)
    parser.add_argument("--base-tokens", type=int, required=True)
    args = parser.parse_args()
    pack(args.normalized_jsonl, args.new_output_dir, args.base_tokens)
