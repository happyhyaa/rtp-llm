"""Independent production 3D supplement; never merge samples with paired 1D/SM."""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from crc_copy_benchmark_stats import TILES, expected_geometry
from run_crc_copy_benchmark import select_cpu, source_provenance


def validate(path, repeat, iterations, correctness_only):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1] == {"type": "complete", "success": True}, "incomplete run"
    meta = records[0]
    assert meta["variants"] == ["copy3d_batch"]
    assert meta["repeat"] == repeat and meta["correctness_only"] == correctness_only
    assert meta["iterations"] == iterations and meta["warmup"] == 30
    assert meta["evict_multiplier"] == 8 and not meta["fallback_allowed"]
    assert meta["seq_size_per_block"] == 128 and meta["tp_size"] == meta["cp_size"] == 1
    assert meta["timing"] == "wall" and meta["regime"] == "cold"
    layouts = {r["layout"]: r for r in records if r["type"] == "layout"}
    assert set(layouts) == set(TILES)
    for name, r in layouts.items():
        assert r["geometry"] == expected_geometry(name)
        assert r["payload_bytes"] == sum(TILES[name])
        assert r["evict_bytes"] >= 8 * meta["l2_bytes"]
        assert r["host_pinned_verified"] and r["source_device_verified"]
    expected = {(d, l, b) for d in ("h2d", "d2h") for l in TILES for b in range(1, 33)}
    checked = []
    samples = collections.defaultdict(dict)
    for r in records:
        if r["type"] not in ("sample", "correctness"):
            continue
        assert r["variant"] == "copy3d_batch"
        key = (r["direction"], r["layout"], r["blocks"])
        assert key in expected
        if r["type"] == "correctness":
            assert r["success"] and r["rotations_checked"] == [0, 1]
            checked.append(key)
        else:
            assert r["repeat"] == repeat and 0 <= r["round"] < iterations
            assert r["round"] not in samples[key], "duplicate sample"
            assert math.isfinite(r["us"]) and r["us"] > 0
            samples[key][r["round"]] = r["us"]
    assert len(checked) == len(expected) and set(checked) == expected
    assert set(samples) == (set() if correctness_only else expected)
    for values in samples.values():
        assert set(values) == set(range(iterations)), "missing sample"
    return meta, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--cpu", default="63")
    args = parser.parse_args()
    provenance = source_provenance()
    cpu = select_cpu(args.cpu)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    env = dict(os.environ, CRC_BENCH_BINARY_SHA256=digest)
    manifest = dict(provenance, binary_sha256=digest, cpu=cpu, status="running",
                    measurement="independent 3D supplement; not paired with previous 1D/SM",
                    ops_construction="inside timed production call", repeats=[80, 81])
    manifest_path = args.output_dir / "run_manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2))
    save()
    merged = collections.defaultdict(list)
    first_meta = None
    try:
        for repeat in (80, 81):
            output = args.output_dir / ("copy3d_%d.jsonl" % repeat)
            cmd = [str(args.binary), "--3d-only", "--output", str(output),
                   "--repeat", str(repeat), "--iterations", "100", "--warmup", "30"]
            if args.correctness_only:
                cmd.append("--correctness-only")
            print("START", repeat, flush=True)
            with output.with_suffix(".stdout").open("x") as out, output.with_suffix(".stderr").open("x") as err:
                result = subprocess.run(cmd, env=env, stdout=out, stderr=err, timeout=3600)
            output.with_suffix(".exitcode").write_text(str(result.returncode))
            if result.returncode:
                raise RuntimeError("3D child failed; see stderr, no retry")
            meta, samples = validate(output, repeat, 100, args.correctness_only)
            assert meta["binary_sha256"] == digest and meta["source_commit"] == provenance["source_commit"]
            if first_meta:
                assert meta["gpu_uuid"] == first_meta["gpu_uuid"]
            first_meta = meta
            for key, values in samples.items():
                merged[key].extend(values.values())
            print("PASS", repeat, "samples", sum(map(len, samples.values())), flush=True)
        rows = []
        for (direction, layout, bs), values in sorted(merged.items()):
            ordered = sorted(values)
            p50 = statistics.median(ordered)
            rows.append(dict(direction=direction, layout=layout, bs=bs, samples=len(values),
                             p50_us=p50, p95_us=ordered[math.ceil(len(ordered)*0.95)-1],
                             throughput_gbps=bs*sum(TILES[layout])/p50/1000))
        if rows:
            with (args.output_dir / "latency.csv").open("x") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (args.output_dir / "analysis.json").write_text(json.dumps(rows, indent=2))
        manifest.update(status="passed", gpu_uuid=first_meta["gpu_uuid"],
                        samples=sum(len(v) for v in merged.values()))
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
