"""Discover BTC Bazel rule declarations in a checked-out revision and build a selected backend."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def discover(repo):
    paths = git(repo, "ls-files", "-z").split("\0")
    for name in paths:
        p = Path(name)
        if p.name not in ("BUILD", "BUILD.bazel") or not (repo / p).is_file():
            continue
        if "block_tree_cache" not in name:
            continue
        text = (repo / p).read_text()
        for rule in re.finditer(r'^\s*name\s*=\s*"([^"]+)"', text, re.M):
            print(f"//{p.parent}:{rule[1]}")


def main(a):
    repo = a.repo.resolve()
    rev = git(repo, "rev-parse", "HEAD").strip()
    if a.action == "locate":
        print(rev)
        discover(repo)
        return
    if not a.target or not a.target.startswith("//") or ":" not in a.target or any(
        c in a.target for c in ("...", "*", "(", ")", " ", "\n")
    ) or a.target.endswith((":all", ":all-targets")):
        raise ValueError("select one explicit native replay binary target")
    options = a.option or []
    def bazel(command, expression, extra=()):
        return subprocess.check_output([a.bazel, command, expression, *options, *extra], cwd=repo, text=True)
    # Bazel resolves macros, select() and transitive dependencies using this checkout and build configuration.
    labels = bazel("cquery", f"deps({a.target})", ["--output=label_kind"]).splitlines()
    bazel("build", a.target)
    outputs = bazel("cquery", a.target, ["--output=files"]).splitlines()
    print(json.dumps(dict(revision=rev, target=a.target, options=options, outputs=outputs,
                          worktree_status=git(repo, "status", "--porcelain").splitlines(),
                          dependencies=labels), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["locate", "build"])
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--target")
    p.add_argument("--bazel", default="bazelisk")
    p.add_argument("--option", action="append")
    main(p.parse_args())
