#!/usr/bin/env python3

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
options, remaining = parser.parse_known_args()


class Dual3DCopyBenchmarkSmokeTest(unittest.TestCase):
    def run_case(self, group, direction, descriptors):
        self.assertTrue(Path(options.binary).is_file(), "dual-path benchmark binary is missing")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            result = subprocess.run(
                [options.binary, "--output", str(output), "--group", group, "--direction", direction,
                 "--descriptors", str(descriptors), "--batch-limit", "500", "--correctness-only"],
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            metadata = records[0]
            self.assertEqual(metadata["group"], group)
            self.assertEqual(metadata["direction"], direction)
            self.assertEqual(metadata["business_count"], 1)
            self.assertEqual(metadata["descriptors"], descriptors)
            self.assertEqual(metadata["batch_limit"], 500)
            self.assertTrue(metadata["gpu_uuid"].startswith("GPU-"))
            self.assertGreater(metadata["production_ops_per_copy_call"], 0)
            self.assertGreater(metadata["benchmark_ops_per_copy_call"], 0)
            correctness = [record for record in records if record.get("type") == "correctness"]
            self.assertEqual({record["variant"] for record in correctness}, {"production_1517", "benchmark_1476"})
            self.assertTrue(all(record["success"] and record["bytes_checked"] > 0 for record in correctness))
            self.assertEqual(sum(record.get("type") == "sample" for record in records), 0)
            self.assertEqual(records[-1], {"type": "complete", "success": True})

    def test_full_d2h_first_descriptor_count(self):
        self.run_case("full", "d2h", 50)

    def test_swa_h2d_batch_limit(self):
        self.run_case("swa", "h2d", 500)


if __name__ == "__main__":
    unittest.main(argv=[__file__, *remaining])
