"""Run independent CPU metadata replay cases, retaining input and outputs."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def row(i, path):
    return dict(
        key_form="prefix",
        block_tokens=256,
        input_len=len(path) * 256 + 128,
        keys=path,
        timestamp_ns=i + 1,
        instance_id="fixture",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("binary", type=Path)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    a.directory = a.directory.resolve()
    a.directory.mkdir(exist_ok=False)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        NVIDIA_VISIBLE_DEVICES="none",
        LOG_LEVEL="ERROR",
    )
    paths = [[100 * j + i for i in range(1, 5)] for j in range(12)]
    cases = [
        (
            "joint",
            {"device_blocks": 64, "swa_blocks": 64, "last_n": 2, "linear_step": 4},
            [[1, 2, 3, 4, 5, 6, 7, 8]] * 2 + [[1, 2, 3, 4, 5, 6, 99]],
        ),
        (
            "force",
            {"swa": False, "device_blocks": 8, "host_blocks": 0, "disk_blocks": 0},
            paths[:3],
        ),
        (
            "capacity_failure",
            {"swa": False, "device_blocks": 3, "host_blocks": 0, "disk_blocks": 0},
            [paths[0], [1001]],
        ),
        ("cp4", {"cp_size": 4, "swa": False}, [list(range(1, 9))] * 2),
        (
            "cp4_swa",
            {"cp_size": 4, "swa": True, "swa_block_tokens": 1024},
            [list(range(1, 9))] * 2,
        ),
        (
            "watermark",
            {
                "swa": False,
                "device_blocks": 16,
                "host_blocks": 16,
                "disk_blocks": 128,
                "device_low": 0.5,
                "device_high": 0.75,
                "host_low": 0.5,
                "host_high": 0.75,
            },
            paths + paths,
        ),
        (
            "timing",
            {
                "device_blocks": 64,
                "swa_blocks": 64,
                "host_blocks": 48,
                "disk_blocks": 128,
                "device_low": 0.64,
                "device_high": 0.80,
                "host_low": 0.64,
                "host_high": 0.80,
                "disk_low": 0.7,
                "disk_high": 0.9,
            },
            [
                list(range(1, 5)) + [10000 + j * 32 + i for i in range(12)]
                for j in ([*range(16), 0, 12, 15, 15] * 50)
            ],
        ),
    ]
    report = {}
    for name, config, requests in cases:
        root = a.directory / name
        root.mkdir()
        save(root / "config.json", config)
        trace = root / "trace.jsonl"
        trace.write_text(
            "".join(json.dumps(row(i, path)) + "\n" for i, path in enumerate(requests))
        )
        with (root / "run.log").open("wb") as log:
            subprocess.run(
                [
                    str(a.binary.resolve()),
                    str(root / "config.json"),
                    str(trace),
                    str(root / "out"),
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=True,
            )
        s = json.loads((root / "out/summary.json").read_text())
        rows = [
            json.loads(x)
            for x in (root / "out/requests.jsonl").read_text().splitlines()
        ]
        assert s["complete"] and len(rows) == s["requests"] == len(requests)
        assert s["hit_tokens"] == sum(x["reuse_tokens"] for x in rows)
        assert s["input_tokens"] == sum(x["input_tokens"] for x in rows)
        assert s["kv_payload_bytes_allocated"] == s["physical_copy_attempts"] == 0
        assert s["failed_requests"] == (1 if name == "capacity_failure" else 0)
        if name == "joint":
            assert [x["reuse_tokens"] for x in rows] == [0, 2048, 1024], rows
        if name in ("cp4", "cp4_swa"):
            assert [x["reuse_tokens"] for x in rows] == [0, 2048], rows
        if name == "force":
            assert (
                s["force_drop_reclaimed_blocks"] > 0
                and s["watermark_scheduled_blocks"] == 0
            )
        if name == "watermark":
            assert (
                s["watermark_scheduled_blocks"] > 0
                and s["device_host"] > 0
                and s["host_disk"] > 0
            ), s
            assert s["disk_device"] > 0 and rows[len(paths)]["reuse_tokens"] == 1024, s
        if name == "capacity_failure":
            assert [x["success"] for x in rows] == [False, True]
        report[name] = {
            "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "summary": s,
        }
        print(
            name,
            s["requests"],
            s["force_drop_reclaimed_blocks"],
            s["watermark_scheduled_blocks"],
            flush=True,
        )
    save(a.directory / "verified.json", report)


if __name__ == "__main__":
    main()
