"""Snapshot configured in-repo inputs to BTC and compare snapshots from two revisions."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

CACHE = "//rtp_llm/cpp/cache"
TARGETS = {
    "btc": CACHE + "/block_tree_cache:block_tree_cache",
    "integration": CACHE + "/block_tree_cache/benchmark:resident_metadata_replay",
}


def snapshot(args):
    files = {}
    package_options = [f"--package_path={args.overlay.resolve()}:%workspace%"] if args.overlay else []

    def commands(command, expression, options):
        return subprocess.check_output(
            [args.bazel, *args.startup_option, command, expression, *package_options, *options],
            cwd=args.repo, text=True,
        ).splitlines()

    def record(label, scope):
        package, name = label[2:].split(":", 1)
        if package == "external":  # Bazel 6 virtual package backed by WORKSPACE.
            return
        path = Path(package) / name
        root = args.repo
        if args.overlay and any((args.overlay / package / build).is_file() for build in ("BUILD", "BUILD.bazel")):
            root = args.overlay
        if path.as_posix() not in files:
            files[path.as_posix()] = {
                "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
                "scope": scope,
            }

    for scope, target in TARGETS.items():
        packages = {}
        rows = commands("cquery", f"deps({target})", ["--output=label_kind", *args.build_option])
        for row in rows:
            # Each label_kind row ends in a label and, for cquery, a configuration.
            labels = [word.lstrip("@") for word in row.split() if word.startswith(("//", "@//", "@@//"))]
            if not labels:
                continue
            label = labels[0]
            packages.setdefault(label.split(":")[0], label)
            if row.startswith("source file "):
                record(label, scope)
        if not packages or not files:
            raise ValueError(f"no configured workspace inputs for {target}")
        # Only follow BUILD/.bzl loads of packages in this configuration.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".query") as query:
            query.write("buildfiles(set(" + " ".join(packages.values()) + "))")
            query.flush()
            for row in commands("query", "--query_file=" + query.name, ["--output=label"]):
                if row.startswith(("//", "@//", "@@//")):
                    record(row.lstrip("@"), scope)
    for name in (".bazelrc", ".bazelversion", "WORKSPACE", "WORKSPACE.bazel", "MODULE.bazel", "MODULE.bazel.lock"):
        if (args.repo / name).is_file():
            record("//:" + name, "build")
    return {"targets": TARGETS, "build_options": args.build_option, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("repo", type=Path)
    snap.add_argument("output", type=Path)
    snap.add_argument("--overlay", type=Path, help="optional Bazel package overlay used by the build")
    snap.add_argument("--bazel", default="bazelisk")
    snap.add_argument("--startup-option", action="append", default=[])
    snap.add_argument("--build-option", action="append", default=[])
    diff = commands.add_parser("diff")
    diff.add_argument("old", type=Path)
    diff.add_argument("new", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        result = snapshot(args)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        return
    old, new = [json.loads(p.read_bytes()) for p in (args.old, args.new)]
    if old["build_options"] != new["build_options"] or old["targets"] != new["targets"]:
        parser.error("compare snapshots from the same targets and build configuration")
    before, after = old["files"], new["files"]
    for path in sorted(before.keys() | after.keys()):
        if before.get(path) != after.get(path):
            status = "added" if path not in before else "removed" if path not in after else "changed"
            print(status, (after.get(path) or before[path])["scope"], path, sep="\t")


if __name__ == "__main__":
    main()
