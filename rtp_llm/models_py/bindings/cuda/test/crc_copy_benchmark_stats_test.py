"""CPU-only regression tests for rejecting incomplete or misleading results."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from crc_copy_benchmark_config import parse_block_counts, resolve_benchmark_config
from crc_copy_benchmark_stats import (
    BLOCKS,
    DIRECTIONS,
    VARIANTS,
    analyze,
    expected_geometry,
    expected_layout,
    read_run,
)


def fixture(
    exclude=False,
    correctness_only=False,
    repeat=80,
    model="flash",
    logical_tokens=1024,
    kernel_tokens=128,
    block_counts=BLOCKS,
):
    config = resolve_benchmark_config(model, logical_tokens, kernel_tokens, block_counts)
    profile = config["profile"]
    block_counts = config["block_counts"]
    shapes = expected_layout(model, logical_tokens, kernel_tokens)
    settings = dict(
        iterations=4,
        warmup=1,
        seed=20260924,
        exclude_1d_h2d=exclude,
        correctness_only=correctness_only,
        model=model,
        logical_tokens_per_block=logical_tokens,
        kernel_tokens_per_block=kernel_tokens,
        block_counts=block_counts,
    )
    meta = dict(
        type="metadata",
        implementation="crc_copy_benchmark_v2",
        model=model,
        variants=list(VARIANTS),
        block_counts=list(block_counts),
        logical_tokens_per_block=logical_tokens,
        kernel_tokens_per_block=kernel_tokens,
        repeat=repeat,
        seed=settings["seed"] + repeat,
        iterations=4,
        warmup=1,
        correctness_only=correctness_only,
        exclude_1d_h2d=exclude,
        evict_multiplier=8,
        timing="wall",
        regime="cold",
        cpu_metadata="warm",
        layout_order="production",
        shape_source="CacheConfigCreator+DeviceBlockPoolConfigHelper",
        source_commit="a" * 40,
        base_commit="b" * 40,
        binary_sha256="c" * 64,
        local_backing_count=True,
        model_num_layers=profile.num_layers,
        hidden_size=profile.hidden_size,
        head_num=profile.head_num,
        indexer_topk=profile.indexer_topk,
        o_groups=profile.o_groups,
        seq_size_per_block=logical_tokens,
        kernel_seq_size_per_block=kernel_tokens,
        cp_size=profile.cp_size,
        tp_size=profile.tp_size,
        cp_mode=profile.cp_mode,
        kv_cache_sharded=profile.kv_cache_sharded,
        gen_num_per_cycle=0,
        fallback_allowed=False,
        l2_bytes=1024,
        driver=13000,
        runtime=13020,
        gpu="fixture",
        gpu_uuid="fixture-uuid",
        sm=103,
    )
    for field in (
        "boundary",
        "production_source",
        "staged_no_crc_source",
        "gather_control",
        "copy3d_stream",
        "copy3d_submit_mutex",
        "host_input",
        "h2d_host_payload",
    ):
        meta[field] = "fixture contract"
    if exclude:
        meta["excluded_cases"] = [
            dict(direction="h2d", variant="copy1d_batch", reason="explicit option")
        ]
    records = [
        meta,
        dict(type="mixed_crc_selftest", variant="integrated_crc", success=True),
    ]
    for layout, expected in shapes.items():
        payload = expected["payload_bytes"]
        records.append(
            dict(
                type="layout",
                layout=layout,
                **{key: value for key, value in expected.items() if key != "offsets"},
                pool_blocks=129,
                source_bytes=payload * 129,
                destination_bytes=payload * 129,
                evict_bytes=8192,
            )
        )
        for direction in DIRECTIONS:
            active = [
                v
                for v in VARIANTS
                if not (exclude and direction == "h2d" and v == "copy1d_batch")
            ]
            for n in block_counts:
                for variant in VARIANTS:
                    record = dict(
                        direction=direction, layout=layout, blocks=n, variant=variant
                    )
                    if variant in active:
                        record.update(
                            type="correctness", rotations_checked=[0, 1], success=True
                        )
                    else:
                        record.update(
                            type="excluded_correctness", reason="explicit option"
                        )
                    records.append(record)
                if not correctness_only:
                    for round_id in range(4):
                        for position, variant in enumerate(active):
                            records.append(
                                dict(
                                    type="sample",
                                    direction=direction,
                                    layout=layout,
                                    blocks=n,
                                    variant=variant,
                                    repeat=repeat,
                                    round=round_id,
                                    position=position,
                                    source_plan=round_id % 2,
                                    timing="wall",
                                    regime="cold",
                                    us=100.0 + n + 10 * position + round_id,
                                )
                            )
        for n in (1, 8, 16, 32):
            records.append(
                dict(
                    type="crc_corruption",
                    layout=layout,
                    blocks=n,
                    variant="integrated_crc",
                    direction="h2d",
                    cases=6,
                    corruption_cases=6,
                    success=True,
                    all_targets_unchanged=True,
                    recovery_success=True,
                    good_bad_good=True,
                    entire_pool_guards=True,
                )
            )
    records.append(dict(type="complete", success=True))
    return settings, records


class BenchmarkStatsTest(unittest.TestCase):
    def test_cli_defaults_and_bs_list_parser(self):
        pro = resolve_benchmark_config("pro")
        flash = resolve_benchmark_config("flash")
        self.assertEqual(
            (pro["logical_tokens_per_block"], pro["kernel_tokens_per_block"]),
            (128, 128),
        )
        self.assertEqual(
            (flash["logical_tokens_per_block"], flash["kernel_tokens_per_block"]),
            (1024, 128),
        )
        self.assertEqual(parse_block_counts("1, 8,16,24,32"), (1, 8, 16, 24, 32))
        for invalid in ("", "1,", "1,,8", "1,x"):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    parse_block_counts(invalid)

    def test_shape_oracle_tracks_model_and_logical_kernel_spans(self):
        expected_baselines = {
            ("pro", 128, 128): (732672, 878160),
            ("flash", 1024, 128): (4087296, 4940160),
        }
        for (model, logical, kernel), payloads in expected_baselines.items():
            with self.subTest(model=model, logical=logical, kernel=kernel):
                shapes = expected_layout(model, logical, kernel)
                self.assertEqual(
                    (shapes["full"]["payload_bytes"], shapes["swa"]["payload_bytes"]),
                    payloads,
                )

        for model in ("pro", "flash"):
            for logical, kernel in (
                (256, 128), (256, 256),
                (1024, 128), (1024, 256), (1024, 1024),
                (4096, 128), (4096, 256), (4096, 1024),
            ):
                with self.subTest(model=model, logical=logical, kernel=kernel):
                    shapes = expected_layout(model, logical, kernel)
                    for shape in shapes.values():
                        self.assertEqual(
                            sum(shape["tile_bytes"]), shape["payload_bytes"]
                        )
                        self.assertEqual(
                            shape["offsets"][-1] + shape["tile_bytes"][-1],
                            shape["payload_bytes"],
                        )

    def test_resolves_both_dsv4_models_and_all_valid_block_spans(self):
        valid_pairs = (
            (256, 128),
            (256, 256),
            (1024, 128),
            (1024, 256),
            (1024, 1024),
            (4096, 128),
            (4096, 256),
            (4096, 1024),
        )
        block_counts = (1, 8, 16, 24, 32)
        for model in ("pro", "flash"):
            for logical_tokens, kernel_tokens in valid_pairs:
                with self.subTest(
                    model=model, logical=logical_tokens, kernel=kernel_tokens
                ):
                    config = resolve_benchmark_config(
                        model, logical_tokens, kernel_tokens, block_counts
                    )
                    self.assertEqual(config["model"], model)
                    self.assertEqual(
                        config["logical_tokens_per_block"], logical_tokens
                    )
                    self.assertEqual(
                        config["kernel_tokens_per_block"], kernel_tokens
                    )
                    self.assertEqual(config["block_counts"], block_counts)

    def test_rejects_invalid_model_span_and_backing_count(self):
        invalid = (
            ("unknown", 1024, 128, (1,)),
            ("pro", 256, 1024, (1,)),
            ("flash", 1024, 192, (1,)),
            ("pro", 1024, 128, (0,)),
            ("flash", 1024, 128, (1, 1)),
            ("pro", 1024, 128, (33,)),
        )
        for model, logical_tokens, kernel_tokens, block_counts in invalid:
            with self.subTest(
                model=model,
                logical=logical_tokens,
                kernel=kernel_tokens,
                blocks=block_counts,
            ):
                with self.assertRaises(ValueError):
                    resolve_benchmark_config(
                        model, logical_tokens, kernel_tokens, block_counts
                    )

    def test_flash_payload_used_for_throughput(self):
        settings, first = fixture(exclude=True)
        _, second = fixture(exclude=True, repeat=81)
        result = analyze([self.write(first, "80.jsonl"), self.write(second, "81.jsonl")], settings)
        shapes = expected_layout("flash", 1024, 128)
        for case in result["cases"]:
            expected_bytes = shapes[case["layout"]]["payload_bytes"]
            for stats in case["variants"].values():
                self.assertAlmostEqual(stats["payload_GBps_at_p50"] * stats["p50_us"] * 1000,
                                       case["blocks"] * expected_bytes, delta=1e-6)

    def test_rejects_wrong_logical_or_kernel_tokens_per_block(self):
        for field, value in (
            ("seq_size_per_block", 128),
            ("kernel_seq_size_per_block", 1024),
        ):
            with self.subTest(field=field):
                settings, records = fixture(exclude=True)
                records[0][field] = value
                with self.assertRaises(ValueError):
                    read_run(self.write(records), settings, 80)

    def test_rejects_pro_parallelism(self):
        settings, records = fixture(exclude=True)
        records[0].update(cp_size=8, tp_size=8, cp_mode="CP_RR")
        with self.assertRaises(ValueError):
            read_run(self.write(records), settings, 80)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def write(self, records, name="raw.jsonl"):
        path = Path(self.temp.name) / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    def test_full_and_explicit_partial_matrices(self):
        for exclude, expected in ((False, 5120), (True, 4608)):
            with self.subTest(exclude=exclude):
                settings, first = fixture(exclude=exclude)
                _, second = fixture(exclude=exclude, repeat=81)
                result = analyze(
                    [self.write(first, "80.jsonl"), self.write(second, "81.jsonl")],
                    settings,
                )
                self.assertEqual(result["sample_count"], expected)
                self.assertEqual(result["paired_round_count"], 1024)
                self.assertEqual(len(result["cases"]), 128)
                cell = result["cases"][0]["variants"]["integrated_crc"]
                self.assertEqual(cell["count"], 8)
                self.assertEqual(cell["p50_us"], 102.5)
                self.assertEqual(cell["p95_us"], 104.0)

    def test_pro_custom_spans_and_bs_subset_are_validated_and_summarized(self):
        counts = (1, 8, 16, 24, 32)
        settings, first = fixture(
            model="pro", logical_tokens=256, kernel_tokens=128, block_counts=counts
        )
        _, second = fixture(
            repeat=81,
            model="pro",
            logical_tokens=256,
            kernel_tokens=128,
            block_counts=counts,
        )
        result = analyze(
            [self.write(first, "80-pro.jsonl"), self.write(second, "81-pro.jsonl")],
            settings,
        )
        self.assertEqual(result["settings"]["block_counts"], counts)
        self.assertEqual(result["sample_count"], 800)
        self.assertEqual(len(result["cases"]), 20)
        full_payload = expected_layout("pro", 256, 128)["full"]["payload_bytes"]
        for case in result["cases"]:
            if case["layout"] == "full":
                stats = case["variants"]["integrated_crc"]
                self.assertAlmostEqual(
                    stats["payload_GBps_at_p50"] * stats["p50_us"] * 1000,
                    case["blocks"] * full_payload,
                    delta=1e-6,
                )

    def test_correctness_only_contains_no_timing(self):
        settings, first = fixture(correctness_only=True)
        _, second = fixture(correctness_only=True, repeat=81)
        result = analyze(
            [self.write(first, "80.jsonl"), self.write(second, "81.jsonl")], settings
        )
        self.assertEqual(result["sample_count"], 0)
        self.assertEqual(result["cases"], [])

    def test_reject_invalid_or_incomplete_evidence(self):
        settings, original = fixture()
        sample = next(i for i, r in enumerate(original) if r["type"] == "sample")
        shape = next(i for i, r in enumerate(original) if r["type"] == "layout")
        corruption = next(
            i for i, r in enumerate(original) if r["type"] == "crc_corruption"
        )
        mutations = {
            "mixed_integer_success": lambda r: r[1].update(success=1),
            "corruption_boolean_blocks": lambda r: r[corruption].update(blocks=True),
            "incomplete": lambda r: r.pop(),
            "integer_success": lambda r: r[-1].update(success=1),
            "boolean_blocks": lambda r: r[sample].update(blocks=True),
            "float_blocks": lambda r: r[sample].update(blocks=1.0),
            "missing_boundary": lambda r: r[0].pop("boundary"),
            "missing_source": lambda r: r[0].pop("source_commit"),
            "invalid_base": lambda r: r[0].update(base_commit="unavailable"),
            "stale_shape_source": lambda r: r[0].update(shape_source="hardcoded"),
            "wrong_tag": lambda r: r[shape]["geometry"][0].update(tag="swa_kv"),
            "wrong_model_layer": lambda r: r[shape]["geometry"][0].update(
                model_layer=60
            ),
            "overlapping_pool": lambda r: r[shape]["geometry"][1].update(
                offset_per_pool_block=0
            ),
            "selected_sentinel": lambda r: r[shape].update(reserved_sentinel_blocks=0),
            "missing_sample": lambda r: r.pop(sample),
            "duplicate_variant": lambda r: r.__setitem__(
                sample + 1, copy.deepcopy(r[sample])
            ),
            "different_source": lambda r: r[sample + 1].update(source_plan=1),
            "duplicate_position": lambda r: r[sample + 1].update(position=0),
            "nonfinite": lambda r: r[sample].update(us=float("nan")),
            "zero_duration": lambda r: r[sample].update(us=0),
            "wrong_shape": lambda r: r[shape].update(payload_bytes=732673),
            "insufficient_eviction": lambda r: r[shape].update(evict_bytes=8191),
            "corruption_wrote_target": lambda r: r[corruption].update(
                all_targets_unchanged=False
            ),
            "silent_exclusion": lambda r: r[0].update(exclude_1d_h2d=True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                records = copy.deepcopy(original)
                mutate(records)
                with self.assertRaises(ValueError):
                    read_run(self.write(records), settings, 80)

    def test_exclusion_cannot_claim_success_or_accept_samples(self):
        settings, original = fixture(exclude=True)
        excluded = next(
            i for i, r in enumerate(original) if r["type"] == "excluded_correctness"
        )
        sample = next(
            i
            for i, r in enumerate(original)
            if r["type"] == "sample" and r["direction"] == "h2d"
        )
        for mutate in (
            lambda r: r[excluded].update(success=True),
            lambda r: r[sample].update(variant="copy1d_batch"),
            lambda r: r.pop(excluded),
        ):
            records = copy.deepcopy(original)
            mutate(records)
            with self.assertRaises(ValueError):
                read_run(self.write(records), settings, 80)

    def test_repeats_must_use_same_gpu(self):
        settings, first = fixture()
        _, second = fixture(repeat=81)
        second[0]["gpu_uuid"] = "other-gpu"
        with self.assertRaisesRegex(ValueError, "gpu_uuid"):
            analyze(
                [self.write(first, "80.jsonl"), self.write(second, "81.jsonl")],
                settings,
            )

    def test_repeats_must_use_same_source_and_binary(self):
        for field in ("source_commit", "base_commit", "binary_sha256"):
            with self.subTest(field=field):
                settings, first = fixture()
                _, second = fixture(repeat=81)
                second[0][field] = "d" * len(second[0][field])
                with self.assertRaisesRegex(ValueError, field):
                    analyze(
                        [self.write(first, "80.jsonl"), self.write(second, "81.jsonl")],
                        settings,
                    )

    def test_provenance_must_match_requested_revision(self):
        settings, records = fixture()
        settings["source_commit"] = "e" * 40
        with self.assertRaisesRegex(ValueError, "source_commit"):
            read_run(self.write(records), settings, 80)


if __name__ == "__main__":
    unittest.main()
