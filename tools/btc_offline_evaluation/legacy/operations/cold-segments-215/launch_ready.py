"""Background supervisor on 215: a new container and empty BTC for each verified segment/policy."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE = "85085895ba19"
EXE = "/work/src/bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/resident_metadata_replay"
FIRST_STAGE = (262510043136, 262497768448, 311133819819)
POLICIES = ("disk-fifo", "all-lru")


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def start(segment, policy):
    label = "segment-{}-{}".format(segment, policy)
    dest = ROOT / label
    name = "REPLAY_USER-cold-1109-" + label
    dest.mkdir(exist_ok=True)
    if (dest / "container.id").exists():
        return
    params = {"disk_fifo": policy == "disk-fifo", "tiers": {
        k: {"full": n, "swa": n} for k, n in
        (("device", 35565), ("host", 166544), ("disk", 3016127))}}
    save(dest / "params.json", params)
    (dest / "authorized-through").write_text("10000000\n")
    args = ["docker", "run", "-d", "--name", name, "--runtime=runc", "--network=none",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user", "{}:{}".format(os.getuid(), os.getgid()), "-e", "CUDA_VISIBLE_DEVICES=",
            "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "LOG_LEVEL=ERROR",
            "-e", "LD_LIBRARY_PATH=/driver:/opt/conda310/lib:/usr/local/cuda/lib64:/usr/lib64",
            "-v", str(ROOT)+":/work:ro", "-v", str(dest)+":/out:rw", "-w", "/out"]
    for lib in ("libcuda.so.1", "libnvidia-ml.so.1"):
        args += ["-v", str((Path("/lib64")/lib).resolve())+":/driver/"+lib+":ro"]
    args += ["--entrypoint", EXE, IMAGE, "/out/params.json", "/work/segment-{}".format(segment), "/out"]
    cid = subprocess.check_output(args, universal_newlines=True).strip()
    (dest / "container.id").write_text(cid+"\n")
    print(label, cid, flush=True)


def run():
    rows = {"1": dict(zip(POLICIES, [100.0*x/FIRST_STAGE[2] for x in FIRST_STAGE[:2]]))}
    while len(rows) < 6:
        for container, marker in (("build","build.ready"), ("input","input.ready")):
            if not (ROOT/marker).exists():
                status = subprocess.check_output(["docker","inspect","--format","{{.State.Status}}",
                    "REPLAY_USER-cold-{}-1109".format(container)], universal_newlines=True).strip()
                if status in ("exited","dead"):
                    raise RuntimeError("{} stopped before {}".format(container, marker))
        for segment in range(2, 7):
            manifest = ROOT / "segment-{}".format(segment) / "segment.json"
            if not (ROOT/"build.ready").exists() or not manifest.is_file():
                continue
            expected = json.loads(manifest.read_text())
            segment_rates = {}
            for policy in POLICIES:
                start(segment, policy)
                dest = ROOT / "segment-{}-{}".format(segment, policy)
                state = dest / "held-10000000.json"
                if state.exists():
                    data = json.loads(state.read_text())
                    if (data["requests"],data["input_tokens"]) != (10000000,expected["input_tokens"]):
                        raise ValueError("segment totals mismatch: {}".format(dest))
                    segment_rates[policy] = 100.0*data["hit_tokens"]/data["input_tokens"]
                elif subprocess.check_output(["docker","inspect","--format","{{.State.Running}}",
                        (dest/"container.id").read_text().strip()], universal_newlines=True).strip() != "true":
                    raise RuntimeError("worker stopped: {}".format(dest))
            if len(segment_rates)==2:
                rows[str(segment)] = segment_rates
        save(ROOT/"cold-results-percent.json",rows)
        if len(rows)<6:
            time.sleep(20)


if __name__ == "__main__":
    with (ROOT/".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
