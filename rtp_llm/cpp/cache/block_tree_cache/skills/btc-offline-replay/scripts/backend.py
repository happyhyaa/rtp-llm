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


def native_recipe(checkout):
    """Derive labels from the actual package containing the native replay entry point."""
    matches = []
    for name in git(checkout, "ls-files", "-z").split("\0"):
        if not name.endswith("/metadata/EvaluationReplayMain.cc"):
            continue
        package = Path(name).parent.parent
        builds = [checkout / package / b for b in ("BUILD.bazel", "BUILD")]
        build = next((p for p in builds if p.is_file()), None)
        if build is None:
            continue
        text = build.read_text()
        rules = set(re.findall(r'^\s*name\s*=\s*"([^"]+)"', text, re.M))
        checks = ("metadata_pool_test", "evaluation_replay_test")
        if {"metadata_no_payload_ops", "evaluation_replay", *checks} <= rules:
            matches.append(dict(target=f"//{package}:evaluation_replay",
                                checks=[f"//{package}:{c}" for c in checks]))
    if len(matches) > 1:
        raise ValueError("multiple native adapters: select the canonical implementation with the source marker")
    return matches[0] if matches else None


FIXED_TARGET = "//tools/btc_offline:replay"
REQUIRED_CHECKS = {"//tools/btc_offline:" + name for name in
                   ("metadata_pool_test", "metadata_replay_test", "evaluation_replay_test")}


def validate_fixed_recipe(recipe):
    if recipe.get("target") != FIXED_TARGET:
        raise ValueError(f"source marker must select {FIXED_TARGET}")
    missing = REQUIRED_CHECKS - set(recipe.get("checks", []))
    if missing:
        raise ValueError("missing native checks: " + ", ".join(sorted(missing)))
    contract = recipe.get("dependency_contract", {})
    for field in ("require", "forbid"):
        entries = contract.get(field)
        if not isinstance(entries, list) or not entries or not all(isinstance(x, str) for x in entries):
            raise ValueError(f"source marker must declare dependency_contract.{field}")


def check_graph(output, contract):
    labels = {m[1] for m in re.finditer(r"^\S+ rule (\S+)", output, re.M)}
    missing = set(contract["require"]) - labels
    forbidden = set(contract["forbid"]) & labels
    if missing or forbidden:
        raise ValueError(f"offline dependency check failed: missing={sorted(missing)}, forbidden={sorted(forbidden)}")


def auto_build(a):
    """Apply the known metadata seams to an isolated checkout, then test that checkout."""
    import os
    import sys
    if not a.work:
        raise ValueError("auto requires a new --work directory for the isolated checkout")
    repo, work = a.repo.resolve(), a.work.resolve()
    rev = git(repo, "rev-parse", "HEAD").strip()
    resources = Path(__file__).resolve().parent.parent / "adapters"
    work.mkdir(parents=True, exist_ok=False)
    checkout = work / "source"
    subprocess.run(["git", "clone", "--no-checkout", "--local", str(repo), str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", rev], check=True)
    marker = checkout / "btc-offline-backend.json"
    fixed = marker.is_file()
    if fixed:
        recipe = json.loads(marker.read_text())
        validate_fixed_recipe(recipe)
    elif not a.legacy_adapter:
        raise ValueError("missing btc-offline-backend.json source marker; "
                         "only use --legacy-adapter for an explicitly requested historical build")
    elif existing := native_recipe(checkout):
        recipe = existing
    else:
        recipe = json.loads((resources / "metadata-replay.json").read_text())
        patch = resources / recipe["patch"]
        test = ["git", "-C", str(checkout), "apply", "--check", str(patch)]
        # A clean reverse check means the exact adapter is already present.
        if subprocess.run(test[:4] + ["--reverse"] + test[4:], capture_output=True).returncode:
            check = subprocess.run(test, capture_output=True, text=True)
            if check.returncode:
                (work / "adapter-conflicts.txt").write_text(check.stderr)
                raise ValueError(f"incompatible metadata seams at {rev}; see {work / 'adapter-conflicts.txt'}")
            subprocess.run(["git", "-C", str(checkout), "apply", str(patch)], check=True)
    package = recipe["target"].split(":")[0].lstrip("/")
    entry = checkout / package / "metadata/EvaluationReplayMain.cc"
    if not fixed and entry.is_file():
        from native_reader import install
        install(entry)
    targets = [recipe["target"], *recipe["checks"]]
    if len(targets) < 2 or any(not re.fullmatch(r"//[\w./+-]+:[\w.+-]+", t) for t in targets):
        raise ValueError("adapter must declare concrete replay and verification targets")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void")
    def run(*args):
        return subprocess.check_output([a.bazel, *args], cwd=checkout, env=env, text=True)
    options = a.option or []
    deps = run("cquery", f"deps({targets[0]})", *options, "--output=label_kind")
    (work / "dependencies.txt").write_text(deps)
    if fixed:
        check_graph(deps, recipe["dependency_contract"])
    run("build", *targets, *options)
    binaries = []
    for t in targets:
        files = run("cquery", t, *options, "--output=files").splitlines()
        if len(files) != 1:
            raise ValueError(f"expected one executable for {t}")
        binaries.append(checkout / files[0])
    for check in binaries[1:]:
        subprocess.run([str(check)], cwd=work, env=env, check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("smoke.py")), str(binaries[0])],
                   cwd=work, env=env, check=True)
    from prepare import digest
    receipt = dict(build_mode="source" if fixed else "legacy-adapter", revision=rev, recipe=recipe, options=options, binary=str(binaries[0]),
                   sha256=digest(binaries[0]), checks_passed=targets[1:])
    (work / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


def main(a):
    if a.action == "auto":
        return auto_build(a)
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
    p.add_argument("action", choices=["locate", "build", "auto"])
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--legacy-adapter", action="store_true", help="opt in to historical adapter discovery/patching for unmarked sources")
    p.add_argument("--target")
    p.add_argument("--work", type=Path)
    p.add_argument("--bazel", default="bazelisk")
    p.add_argument("--option", action="append")
    main(p.parse_args())
