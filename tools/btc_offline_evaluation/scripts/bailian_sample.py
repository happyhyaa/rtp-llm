"""Convert a bounded, stable-ordered sample of Bailian request rows at their base granularity."""

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from decimal import Decimal
from itertools import islice
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--prefix-module", type=Path, required=True)
    p.add_argument("--hash-mode", choices=["raw", "prefix"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not 0 < a.limit <= 100000 or not a.domain:
        raise ValueError("Provide a bounded sample size and explicit cache domain")
    a.output.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location("upstream_hash", a.prefix_module)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    rows = []
    raw_hash = hashlib.sha256()
    with subprocess.Popen(
        ["zstd", "-dc", "--", str(a.source)], stdout=subprocess.PIPE
    ) as proc:
        try:
            for line_no, line in enumerate(islice(proc.stdout, a.limit), 1):
                raw_hash.update(line)
                r = json.loads(line, parse_float=Decimal)
                length = r["input_length"]
                raw = r["input_block_hash_ids"]
                if (
                    type(length) != int
                    or length <= 0
                    or not isinstance(raw, list)
                    or len(raw) not in {length // 256, (length + 255) // 256}
                ):
                    raise ValueError(f"bad dimensions at {line_no}")
                if any(
                    not isinstance(k, str) or not re.fullmatch("[0-9a-fA-F]{16}", k)
                    for k in raw
                ):
                    raise ValueError(f"bad hash at {line_no}")
                keys = [int(k, 16) for k in raw[: length // 256]]
                keys = [k if k < 1 << 63 else k - (1 << 64) for k in keys]
                if a.hash_mode == "raw":
                    keys = upstream.apply_prefix_hash(keys)
                t = Decimal(str(r["timestamp"])) * 1000000000
                if (
                    t <= 0
                    or t != t.to_integral_value()
                    or not isinstance(r["request_id"], str)
                ):
                    raise ValueError("invalid timestamp or request identity")
                rows.append(
                    dict(
                        request_id=r["request_id"],
                        timestamp_ns=int(t),
                        instance_id=a.domain,
                        input_len=length,
                        block_tokens=256,
                        key_form="prefix",
                        keys=keys,
                        source_line=line_no,
                    )
                )
        finally:
            proc.stdout.close()
            proc.wait()
        if len(rows) != a.limit or proc.returncode not in (0, -13, 141):
            raise RuntimeError(
                f"short/corrupt sample: {len(rows)} rows, zstd exit {proc.returncode}"
            )
    rows.sort(key=lambda r: (r["timestamp_ns"], r["source_line"]))
    trace = a.output / "trace.jsonl"
    with trace.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    (a.output / "input-manifest.json").write_text(
        json.dumps(
            dict(
                source=str(a.source),
                source_prefix_sha256=raw_hash.hexdigest(),
                limit=a.limit,
                domain=a.domain,
                ordering="first N file rows, stable sorted by timestamp/source_line",
                base_tokens=256,
                original_key_form=a.hash_mode,
                external_cp_coarsening=False,
                prefix_module_sha256=digest(a.prefix_module),
                trace_sha256=digest(trace),
                requests=len(rows),
                input_tokens=sum(r["input_len"] for r in rows),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
