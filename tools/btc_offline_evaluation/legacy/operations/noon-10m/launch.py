"""Start seven independently resumable CPU workers from the verified 24h cohort."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

base = Path("/data0/REPLAY_USER/hit-rate-evaluation")
run = base / "noon-a3fdca664"
previous = base / "day24-10m-20261106"
exe = run / "source/bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/resident_metadata_replay"
old = json.loads((previous / "btc/state/launch.json").read_text())
budget = json.loads((previous / "btc/finite-1to1-capacity.json").read_text())
image = old["image"]
prefix = old["argv"][:old["argv"].index("--entrypoint")]
groups = [("control-fifo", True, None), ("all-lru", False, None)]
groups.extend((f"fifo-last-{n}", True, n) for n in (1, 5, 10, 15, 20))
binary_hash = hashlib.sha256(exe.read_bytes()).hexdigest()
for group, fifo, tail in groups:
    dest = run / group
    dest.mkdir(exist_ok=True)
    if (dest / "container.id").exists():
        continue
    settings = {**budget, "disk_fifo": fifo}
    if tail is not None:
        settings["last_n"] = tail
        settings["tiers"] = json.loads((run / "qps-tiers.json").read_text())[str(tail)]
        settings["ratio"] = "Device SWA=1000*n; Host and Disk follow Device FULL:SWA"
    cfg = dest / "params.json"
    cfg.write_text(json.dumps(settings, indent=2))
    (dest / "authorized-through").write_text("10000000\n")
    args = list(prefix)
    args[args.index("--name") + 1] = f"REPLAY_USER-noon-a3fdca664-{group}"
    args[args.index(f"{previous}/btc/state:/out:rw")] = f"{dest}:/out:rw"
    args += ["--entrypoint", str(exe), image, str(cfg), str(previous / "input"), "/out"]
    manifest = dict(binary_sha256=binary_hash, commit="a3fdca664089a44769b1d8c220ea0141fd55731f",
                    input_sha256=old["input_sha256"], argv=args, policy=settings)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    container = subprocess.check_output(args, text=True).strip()
    (dest / "container.id").write_text(container + "\n")
    print(group, container, flush=True)
