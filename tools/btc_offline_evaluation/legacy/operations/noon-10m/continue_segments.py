"""Run on 110 in noon-a3fdca664, using resident sessions and published .btcbin chunks."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

GROUPS = ("control-fifo", "all-lru")
STEP = 10_000_000
END = 6 * STEP


def state(path):
    return json.loads(path.read_text())


def check_container(group):
    name = f"REPLAY_USER-noon-a3fdca664-{group.name}"
    cid, running = subprocess.check_output(
        ["docker", "inspect", "--format", "{{.Id}} {{.State.Running}}", name], text=True
    ).split()
    if running != "true" or cid != (group / "container.id").read_text().strip():
        raise RuntimeError(f"Original resident process unavailable: {name}")


def table(groups):
    print("| 请求区间 | Disk FIFO | 分层 LRU |", flush=True)
    print("|---|---:|---:|", flush=True)
    previous = [(0, 0)] * len(groups)
    for end in range(STEP, END + 1, STEP):
        rates, totals = [], []
        for i, group in enumerate(groups):
            path = group / f"held-{end}.json"
            if not path.exists():
                rates.append("待测")
                continue
            record = state(path)
            assert record["requests"] == end and record["next_ordinal"] == end + 1
            hits, tokens = record["hit_tokens"], record["input_tokens"]
            delta = tokens - previous[i][1]
            totals.append(delta)
            rates.append(f"{100 * (hits - previous[i][0]) / delta:.5f}%")
            previous[i] = hits, tokens
        if len(set(totals)) > 1:
            raise ValueError(f"Different input token totals for segment ending {end}")
        print(f"| {end-STEP+1:,}–{end:,} | " + " | ".join(rates) + " |", flush=True)


def advance(groups, source):
    for end in range(2 * STEP, END + 1, STEP):
        for group in groups:
            check_container(group)
        if not all((source / f"{i:06d}.btcbin").is_file()
                   for i in range((end-STEP)//100000, end//100000)):
            subprocess.run(["python3", str(source / "continue_input.py"), "--target-requests", str(end)], check=True)
        for group in groups:
            if (group / f"held-{end}.json").exists():
                continue
            prior = state(group / f"held-{end-STEP}.json")
            target = group / "authorized-through"
            current = int(target.read_text())
            if prior["next_ordinal"] != end-STEP+1 or current not in (end-STEP, end):
                raise RuntimeError(f"Unexpected continuation boundary in {group}")
            tmp = target.with_suffix(".segment-tmp")
            tmp.write_text(f"{end}\n")
            os.replace(tmp, target)
        while not all((g / f"held-{end}.json").exists() for g in groups):
            for group in groups:
                check_container(group)
            time.sleep(30)
        table(groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advance", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    groups = [root / g for g in GROUPS]
    table(groups)
    if args.advance:
        with (root / ".segments.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            advance(groups, root.parent / "day24-10m-20261106/input")
