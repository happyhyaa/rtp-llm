"""Reject incomplete, duplicate, fallback and invalid 3D supplement samples."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from crc_copy_benchmark_stats_test import fixture
from run_3d_copy_supplement import validate


class SupplementTest(unittest.TestCase):
    def records(self, correctness=False):
        _, records = fixture(correctness_only=correctness)
        records = [r for r in records if r.get("variant") != "staged_no_crc"]
        records[0]["variants"] = ["copy3d_batch"]
        records[0]["warmup"] = 30
        for r in records:
            if "variant" in r:
                r["variant"] = "copy3d_batch"
        return records

    def check(self, records, correctness=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            return validate(path, 80, 4, correctness)

    def test_complete_matrix(self):
        _, samples = self.check(self.records())
        self.assertEqual(len(samples), 128)
        self.assertEqual(sum(map(len, samples.values())), 512)

    def test_correctness_only(self):
        self.assertEqual(self.check(self.records(True), True)[1], {})

    def test_reject_corruption(self):
        original = self.records()
        index = next(i for i, r in enumerate(original) if r["type"] == "sample")
        mutations = [lambda r: r.pop(index), lambda r: r.insert(index, copy.deepcopy(r[index])),
                     lambda r: r[index].update(variant="generic"),
                     lambda r: r[index].update(us=float("nan")),
                     lambda r: r.pop(), lambda r: r[0].update(evict_multiplier=1)]
        for mutate in mutations:
            records = copy.deepcopy(original)
            mutate(records)
            with self.assertRaises(AssertionError):
                self.check(records)


if __name__ == "__main__":
    unittest.main()
