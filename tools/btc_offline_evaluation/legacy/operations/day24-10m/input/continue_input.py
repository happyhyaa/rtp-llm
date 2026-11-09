"""Append verified base-256 chunks through an explicitly selected cumulative day ordinal."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
PREP = BASE / "last-n-20261029-v1/prepared"
REF = BASE / "sgp04-cp4-b256-20261102-v3/input1024"
DAY = ROOT / "bailian-20261027-24h.json"
FIELDS = ("timestamp_ns", "file_index", "source_line")
CHUNK_SIZE = 100000


def read_json(path):
    return json.loads(path.read_bytes())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cursor(row):
    return [row[key] for key in FIELDS]


def initial_offsets(state):
    """Locate the last approved row in raw and reference metadata by its stable cursor."""
    last = cursor(state["last"])
    with (PREP / f"{last[1]:03}.sorted.tsv").open("rb") as f:
        source_offset = 0
        for line in f:
            current = list(map(int, line.split(b"\t", 3)[:3]))
            if current >= last:
                if current != last:
                    raise ValueError("missing initial source cursor")
                break
            source_offset += len(line)
        else:
            raise ValueError("source cursor past EOF")
    with (REF / "requests.meta.tsv").open("rb") as f:
        for _ in range(state["requests"] - 1):
            if not f.readline():
                raise ValueError("short initial metadata")
        meta_offset = f.tell()
        if list(map(int, f.readline().split(b"\t", 3)[:3])) != last:
            raise ValueError("initial metadata ordinal/cursor mismatch")
    return dict(state, source_offset=source_offset, meta_offset=meta_offset)


def check_chunk_size(receipt, day_requests):
    end = receipt["requests"]
    start = end - receipt["chunk_requests"]
    if (start % CHUNK_SIZE or not 0 <= start < end <= day_requests
            or receipt["chunk_requests"] != min(CHUNK_SIZE, day_requests - start)):
        raise ValueError("continuation chunk must be full except at the end of the Beijing day")


def publish(staging, receipt):
    chunk = ROOT / receipt["file"]
    temporary = staging / "chunk.tmp"
    permanent = staging / (chunk.stem + ".json")
    if not chunk.exists():
        if sha(temporary) != receipt["sha256"]:
            raise ValueError("staged SHA-256 mismatch")
        os.link(temporary, chunk)
    elif sha(chunk) != receipt["sha256"]:
        raise ValueError(f"conflicting existing chunk {chunk}")
    os.replace(staging / "pending.json", permanent)
    temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-requests", type=int, required=True,
                        help="cumulative request count, aligned to 100000 or exactly the Beijing day total")
    args = parser.parse_args()
    day = read_json(DAY)
    baseline = read_json(ROOT / "cohort.json")
    if not baseline["requests"] < args.target_requests <= day["requests"]:
        parser.error(f"target must exceed {baseline['requests']} and be at most {day['requests']}")
    if args.target_requests != day["requests"] and args.target_requests % CHUNK_SIZE:
        parser.error("target must be a multiple of 100000 or exactly the Beijing day total")
    baseline_audit = read_json(ROOT / "verification.json")
    if baseline_audit["cohort"] != baseline:
        raise ValueError("initial cohort/audit mismatch")
    last = baseline_audit["chunks"][-1]
    if sha(ROOT / last["file"]) != last["sha256"]:
        raise ValueError("initial predecessor chunk changed")
    subprocess.run([sys.executable, ROOT/"preflight.py", ROOT, PREP, REF, DAY], check=True)
    staging = ROOT / "continuation"
    staging.mkdir(exist_ok=True)
    with (staging / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pending = staging / "pending.json"
        if pending.exists():
            receipt = read_json(pending)
            check_chunk_size(receipt, day["requests"])
            publish(staging, receipt)
        state = baseline
        index = baseline["verified_chunks"]
        for path in sorted(staging.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json")):
            existing = read_json(path)
            check_chunk_size(existing, day["requests"])
            if (path.stem != f"{index:06}" or existing["file"] != f"{index:06}.btcbin"
                    or existing["predecessor"] != state["last"] or existing["first"] != state["next"]
                    or existing["requests"] != state["requests"] + existing["chunk_requests"]
                    or existing["input_tokens"] != state["input_tokens"] + existing["chunk_input_tokens"]
                    or sha(ROOT / existing["file"]) != existing["sha256"]):
                raise ValueError(f"existing continuation receipt mismatch {path}")
            state = existing
            index += 1
        if state["requests"] > args.target_requests:
            raise ValueError("target is below already prepared request count")
        if state is baseline:
            state = initial_offsets(state)
        while state["requests"] < args.target_requests:
            count = min(CHUNK_SIZE, args.target_requests-state["requests"])
            temporary = staging / "chunk.tmp"
            if temporary.exists():
                # A recovered hard link may still reference the published immutable chunk.
                temporary.unlink()
            output = ROOT / f"{index:06}.btcbin"
            if output.exists():
                raise ValueError(f"untracked chunk {output}")
            params = [PREP, REF, temporary, day["requests"], state["requests"], count, state["input_tokens"],
                      state["last"]["file_index"], state["source_offset"], state["meta_offset"],
                      *cursor(state["last"]), *cursor(state["next"]), day["input_tokens"]]
            result = subprocess.check_output([str(ROOT/"append_chunk"), *map(str, params)])
            receipt = dict(json.loads(result), file=output.name, sha256=sha(temporary),
                           predecessor=state["last"], first=state["next"])
            check_chunk_size(receipt, day["requests"])
            draft = staging / "receipt.tmp"
            draft.write_text(json.dumps(receipt, indent=2) + "\n")
            os.replace(draft, pending)
            publish(staging, receipt)
            state = receipt
            index += 1
            print(output.name, state["requests"], state["input_tokens"], flush=True)
        print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
