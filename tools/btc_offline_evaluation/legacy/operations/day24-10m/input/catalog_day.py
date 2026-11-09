"""Verify the complete BTCPFX01 day and publish an ordered, self-contained chunk catalog."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def build_catalog(root):
    day = json.loads((root / "bailian-20261027-24h.json").read_bytes())
    baseline = json.loads((root / "verification.json").read_bytes())["chunks"]
    chunks = []
    for index, path in enumerate(sorted(root.glob("*.btcbin"))):
        if path.name != f"{index:06}.btcbin":
            raise ValueError(f"missing or unordered chunk before {path}")
        receipt = baseline[index] if index < len(baseline) else json.loads(
            (root / "continuation" / f"{index:06}.json").read_bytes()
        )
        with path.open("rb") as source:
            header = source.read(38)
            magic, base, count, tokens, size, domain = struct.unpack("<8sIQQI6s", header)
            if (magic, base, size, domain) != (b"BTCPFX01", 256, 6, b"global"):
                raise ValueError(f"invalid header in {path}")
            if count != min(100000, day["requests"] - index * 100000):
                raise ValueError(f"invalid request count in {path}")
            digest = hashlib.sha256(header)
            for block in iter(lambda: source.read(8 << 20), b""):
                digest.update(block)
        if receipt["file"] != path.name or digest.hexdigest() != receipt["sha256"]:
            raise ValueError(f"identity mismatch for {path}")
        chunks.append(dict(file=path.name, sha256=digest.hexdigest(), requests=count,
                           input_tokens=tokens, bytes=path.stat().st_size))
    for field in ("requests", "input_tokens"):
        if sum(row[field] for row in chunks) != day[field]:
            raise ValueError(f"daily {field} mismatch")
    return dict(format="BTCPFX01", base_tokens=256, key_type="prefix", cp_applied=False,
                timezone="Asia/Shanghai", date="2026-10-27", requests=day["requests"],
                input_tokens=day["input_tokens"], bytes=sum(c["bytes"] for c in chunks),
                chunks=chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    root = parser.parse_args().input
    catalog = build_catalog(root)
    temporary = root / "day-catalog.json.tmp"
    temporary.write_text(json.dumps(catalog, indent=2) + "\n")
    temporary.replace(root / "day-catalog.json")
