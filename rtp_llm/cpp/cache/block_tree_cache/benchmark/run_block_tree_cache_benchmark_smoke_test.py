#!/usr/bin/env python3
"""End-to-end GPU smoke for the BlockTreeCache benchmark harness."""

import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from unittest.mock import patch


def _runfiles_path(*parts):
    root = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
    if not root:
        return None
    for workspace in ("rtp_llm", "github-opensource"):
        candidate = os.path.join(root, workspace, *parts)
        if os.path.exists(candidate):
            return candidate
    return None


def _libpython_dir():
    lib_dir = sysconfig.get_config_var("LIBDIR")
    runtime_names = []
    for key in ("INSTSONAME", "LDLIBRARY"):
        name = sysconfig.get_config_var(key)
        if name and ".so" in name:
            runtime_names.append(name)
    runtime_names.append(
        f"libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0"
    )
    runtime_names = list(dict.fromkeys(runtime_names))

    if lib_dir and any(
        os.path.exists(os.path.join(lib_dir, name)) for name in runtime_names
    ):
        return lib_dir

    root = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
    if not root:
        return None
    for base, _, files in os.walk(root):
        if any(name in files for name in runtime_names):
            return base
    return None


def _load_json(path):
    with open(path) as source:
        return json.load(source)


class BenchmarkSmokeTest(unittest.TestCase):
    def test_libpython_dir_prefers_interpreter_libdir_with_runtime_soname(self):
        with tempfile.TemporaryDirectory() as tmp:
            version = f"{sys.version_info.major}.{sys.version_info.minor}"
            runtime_name = f"libpython{version}.so.1.0"
            runfiles_dir = os.path.join(tmp, "runfiles")
            runfiles_lib = os.path.join(runfiles_dir, "workspace", "lib")
            interpreter_lib = os.path.join(tmp, "interpreter_lib")
            os.makedirs(runfiles_lib)
            os.makedirs(interpreter_lib)
            open(os.path.join(runfiles_lib, f"libpython{version}.so"), "wb").close()
            open(os.path.join(interpreter_lib, runtime_name), "wb").close()

            values = {
                "LIBDIR": interpreter_lib,
                "INSTSONAME": "libpython3.10.a",
                "LDLIBRARY": "libpython3.10.a",
            }
            with patch.dict(
                os.environ, {"RUNFILES_DIR": runfiles_dir}, clear=False
            ), patch.object(
                sysconfig, "get_config_var", side_effect=values.get
            ):
                self.assertEqual(_libpython_dir(), interpreter_lib)

    def test_libpython_dir_falls_back_to_versioned_runfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_name = (
                f"libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0"
            )
            runfiles_lib = os.path.join(tmp, "workspace", "lib")
            os.makedirs(runfiles_lib)
            open(os.path.join(runfiles_lib, runtime_name), "wb").close()

            values = {"LIBDIR": None, "INSTSONAME": None, "LDLIBRARY": None}
            with patch.dict(
                os.environ, {"RUNFILES_DIR": tmp}, clear=False
            ), patch.object(sysconfig, "get_config_var", side_effect=values.get):
                self.assertEqual(_libpython_dir(), runfiles_lib)

    def test_libpython_dir_rejects_unversioned_linker_stub(self):
        with tempfile.TemporaryDirectory() as tmp:
            version = f"{sys.version_info.major}.{sys.version_info.minor}"
            runfiles_lib = os.path.join(tmp, "workspace", "lib")
            os.makedirs(runfiles_lib)
            open(os.path.join(runfiles_lib, f"libpython{version}.so"), "wb").close()

            values = {"LIBDIR": None, "INSTSONAME": None, "LDLIBRARY": None}
            with patch.dict(
                os.environ, {"RUNFILES_DIR": tmp}, clear=False
            ), patch.object(sysconfig, "get_config_var", side_effect=values.get):
                self.assertIsNone(_libpython_dir())

    def test_smoke_suite(self):
        driver = _runfiles_path(
            "rtp_llm/cpp/cache/block_tree_cache/benchmark/run_block_tree_cache_benchmark.py"
        )
        self.assertIsNotNone(driver, "driver script not found in runfiles")

        env = dict(os.environ)
        lib_dir = _libpython_dir()
        if lib_dir:
            env["LD_LIBRARY_PATH"] = (
                lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            )
        env["BLOCK_TREE_CACHE_BENCHMARK_TEST_CONFIG"] = "1"

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "out")
            cmd = [
                sys.executable,
                driver,
                "--suite",
                "smoke",
                "--output-dir",
                output_dir,
                "--perf",
                "off",
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"benchmark smoke suite failed rc={proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
            )

            results = {}
            for case in ("smoke_tree_online_mini", "smoke_transfer_d2h_mini"):
                case_dir = os.path.join(output_dir, "smoke", case)
                manifest_path = os.path.join(case_dir, "manifest.json")
                result_path = os.path.join(case_dir, "rep_0000", "result.json")
                self.assertTrue(
                    os.path.exists(manifest_path), f"missing {case} manifest"
                )
                self.assertTrue(os.path.exists(result_path), f"missing {case} result")
                self.assertEqual(_load_json(manifest_path).get("status"), "completed")
                results[case] = _load_json(result_path)
                self.assertEqual(results[case].get("status"), "completed")

            tree = results["smoke_tree_online_mini"]
            self.assertGreater(tree.get("metrics", {}).get("loads_committed", 0), 0)

            transfer = results["smoke_transfer_d2h_mini"]
            self.assertGreater(
                transfer.get("workload", {}).get("succeeded_operations", 0), 0
            )


if __name__ == "__main__":
    unittest.main()
