"""Validate the complete copy benchmark matrix before reporting any timing."""

import collections
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
from pathlib import Path

from crc_copy_benchmark_config import MAX_BACKINGS, model_profile, resolve_benchmark_config

VARIANTS = (
    "integrated_crc",
    "copy1d_batch",
    "copy3d_batch",
    "staged_no_crc",
    "gather_control",
)
DIRECTIONS = ("d2h", "h2d")
BLOCKS = tuple(range(1, MAX_BACKINGS + 1))


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _profile_groups(model, layout):
    profile = model_profile(model)
    tags_by_ratio = {
        4: ("csa_kv", "indexer_kv", "indexer_state", "csa_state", "swa_kv"),
        128: ("hca_kv", "hca_state", "swa_kv"),
        0: ("swa_kv",),
    }
    selected_tags = (
        {"csa_kv", "indexer_kv", "hca_kv"}
        if layout == "full"
        else {"swa_kv", "indexer_state", "csa_state"}
    )
    order = []
    for ratio in profile.ratios:
        for tag in tags_by_ratio[ratio]:
            if tag in selected_tags and tag not in order:
                order.append(tag)
    layer_ids = {
        "csa_kv": [i for i, ratio in enumerate(profile.ratios) if ratio == 4],
        "indexer_kv": [i for i, ratio in enumerate(profile.ratios) if ratio == 4],
        "hca_kv": [i for i, ratio in enumerate(profile.ratios) if ratio == 128],
        "swa_kv": list(range(profile.num_layers)),
        "indexer_state": [i for i, ratio in enumerate(profile.ratios) if ratio == 4],
        "csa_state": [i for i, ratio in enumerate(profile.ratios) if ratio == 4],
    }
    return profile, order, layer_ids


def expected_layout(model, logical_tokens_per_block, kernel_tokens_per_block):
    config = resolve_benchmark_config(
        model, logical_tokens_per_block, kernel_tokens_per_block, (1,)
    )
    profile = config["profile"]
    logical = config["logical_tokens_per_block"]
    kernel = config["kernel_tokens_per_block"]
    pages_per_logical = logical // kernel
    cp = profile.cp_size

    widths = {
        "csa_kv": _align((kernel // 4) * 584, 576) * pages_per_logical,
        "indexer_kv": (kernel // 4) * 132 * pages_per_logical,
        "hca_kv": _align((kernel // 128) * 584, 576) * pages_per_logical,
        "swa_kv": _align(128 * 584, 576) // cp,
        "indexer_state": (8 // cp if cp > 1 else 8) * 128 * 4 * 4,
        "csa_state": (8 // cp if cp > 1 else 8) * 512 * 4 * 4,
    }
    result = {}
    for layout in ("full", "swa"):
        _, order, layer_ids = _profile_groups(model, layout)
        geometry, tile_bytes, offsets = [], [], []
        offset = 0
        for member, tag in enumerate(order):
            for local_layer, model_layer in enumerate(layer_ids[tag]):
                width = widths[tag]
                geometry.append(
                    dict(
                        tag=tag,
                        member=member,
                        local_layer=local_layer,
                        model_layer=model_layer,
                        bytes=width,
                        offset_per_pool_block=offset,
                    )
                )
                tile_bytes.append(width)
                offsets.append(offset)
                offset += width
        result[layout] = {
            "tile_bytes": tile_bytes,
            "geometry": geometry,
            "offsets": offsets,
            "tiles": len(tile_bytes),
            "payload_bytes": sum(tile_bytes),
        }

    max_payload = max(shape["payload_bytes"] for shape in result.values())
    for shape in result.values():
        payload = shape["payload_bytes"]
        encoded = _align(payload + 4, 16)
        shape.update(
            reserved_sentinel_blocks=1,
            encoded_bytes=encoded,
            host_stride=_align(encoded, 4096),
            staging_stride=_align(max_payload + 4, 16),
            host_pinned_verified=True,
            source_device_verified=True,
            copy3d_operations_per_backing=len(order),
        )
    return result


def expected_geometry(layout, model="flash", logical_tokens_per_block=1024, kernel_tokens_per_block=128):
    return expected_layout(model, logical_tokens_per_block, kernel_tokens_per_block)[layout]["geometry"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            exact(actual[k], v) for k, v in expected.items()
        )
    return actual == expected


def case_key(record, blocks=BLOCKS):
    require(
        type(record["blocks"]) is int and record["blocks"] in blocks,
        "Invalid backing count",
    )
    return record["direction"], record["layout"], record["blocks"], record["variant"]


def active_variants(direction, exclude_1d_h2d):
    return tuple(
        v
        for v in VARIANTS
        if not (exclude_1d_h2d and direction == "h2d" and v == "copy1d_batch")
    )


def read_run(path, settings, repeat):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    require(
        records
        and records[0].get("type") == "metadata"
        and exact(records[-1], {"type": "complete", "success": True}),
        f"Incomplete benchmark: {path}",
    )
    meta = records[0]
    exclude = settings["exclude_1d_h2d"]
    config = resolve_benchmark_config(
        settings["model"],
        settings["logical_tokens_per_block"],
        settings["kernel_tokens_per_block"],
        settings["block_counts"],
    )
    profile = config["profile"]
    blocks = list(config["block_counts"])
    shapes = expected_layout(
        config["model"],
        config["logical_tokens_per_block"],
        config["kernel_tokens_per_block"],
    )
    expected_meta = {
        "implementation": "crc_copy_benchmark_v2",
        "variants": list(VARIANTS),
        "model": config["model"],
        "block_counts": blocks,
        "logical_tokens_per_block": config["logical_tokens_per_block"],
        "kernel_tokens_per_block": config["kernel_tokens_per_block"],
        "repeat": repeat,
        "seed": settings["seed"] + repeat,
        "iterations": settings["iterations"],
        "warmup": settings["warmup"],
        "correctness_only": settings["correctness_only"],
        "exclude_1d_h2d": exclude,
        "evict_multiplier": 8,
        "timing": "wall",
        "regime": "cold",
        "cpu_metadata": "warm",
        "layout_order": "production",
        "shape_source": "CacheConfigCreator+DeviceBlockPoolConfigHelper",
        "local_backing_count": True,
        "seq_size_per_block": config["logical_tokens_per_block"],
        "kernel_seq_size_per_block": config["kernel_tokens_per_block"],
        "model_num_layers": profile.num_layers,
        "hidden_size": profile.hidden_size,
        "head_num": profile.head_num,
        "indexer_topk": profile.indexer_topk,
        "o_groups": profile.o_groups,
        "cp_size": profile.cp_size,
        "tp_size": profile.tp_size,
        "cp_mode": profile.cp_mode,
        "kv_cache_sharded": profile.kv_cache_sharded,
        "gen_num_per_cycle": 0,
        "fallback_allowed": False,
    }
    for key, value in expected_meta.items():
        require(
            exact(meta.get(key), value),
            f"Metadata mismatch: {key}",
        )
    for key, length in (
        ("source_commit", 40),
        ("base_commit", 40),
        ("binary_sha256", 64),
    ):
        require(
            isinstance(meta.get(key), str)
            and re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", meta[key]) is not None,
            f"Missing or invalid provenance: {key}",
        )
        if key in settings:
            require(meta[key] == settings[key], f"Provenance mismatch: {key}")
    for key in ("l2_bytes", "driver", "runtime", "sm"):
        require(type(meta[key]) is int and meta[key] > 0, f"Invalid metadata: {key}")
    for key in (
        "boundary",
        "production_source",
        "staged_no_crc_source",
        "gather_control",
        "copy3d_stream",
        "copy3d_submit_mutex",
        "host_input",
        "h2d_host_payload",
    ):
        require(
            isinstance(meta.get(key), str) and bool(meta[key]),
            f"Missing measurement contract: {key}",
        )
    require(meta["driver"] >= 13000 and meta["runtime"] >= 13000, "CUDA 13 required")
    require(bool(meta["gpu_uuid"]), "Missing GPU identity")
    exclusions = meta.get("excluded_cases", [])
    require(len(exclusions) == int(exclude), "Unexpected exclusion metadata")
    if exclude:
        require(
            exclusions[0]["direction"] == "h2d"
            and exclusions[0]["variant"] == "copy1d_batch"
            and bool(exclusions[0].get("reason")),
            "Exclusion must explicitly identify 1D H2D",
        )

    matrix = {
        (d, layout, n, v)
        for d in DIRECTIONS
        for layout in shapes
        for n in blocks
        for v in active_variants(d, exclude)
    }
    rounds = 0 if settings["correctness_only"] else settings["iterations"]
    expected_counts = {
        "metadata": 1,
        "layout": 2,
        "mixed_crc_selftest": 1,
        "correctness": len(matrix),
        "crc_corruption": 8,
        "complete": 1,
    }
    if rounds:
        expected_counts["sample"] = len(matrix) * rounds
    if exclude:
        expected_counts["excluded_correctness"] = 2 * len(blocks)
    counts = collections.Counter(r["type"] for r in records)
    require(counts == collections.Counter(expected_counts), f"Record counts: {counts}")
    layouts = {r["layout"]: r for r in records if r["type"] == "layout"}
    require(set(layouts) == set(shapes), "Layout coverage")
    for layout, expected in shapes.items():
        shape = layouts[layout]
        payload = expected["payload_bytes"]
        for key, value in expected.items():
            if key == "offsets":
                continue
            require(exact(shape.get(key), value), f"Invalid {layout} shape: {key}")
        for key in ("source_bytes", "destination_bytes", "pool_blocks", "evict_bytes"):
            require(
                type(shape[key]) is int and shape[key] > 0, f"Invalid pool field: {key}"
            )
        require(
            shape["pool_blocks"] >= 129 and shape["evict_bytes"] % 16 == 0,
            "Insufficient pool or unaligned eviction capacity",
        )
        require(
            shape["source_bytes"]
            == shape["destination_bytes"]
            == shape["pool_blocks"] * payload,
            "Pool capacity mismatch",
        )
        require(
            shape["source_bytes"] >= 8 * meta["l2_bytes"]
            and (shape["pool_blocks"] - 1) * payload >= 8 * meta["l2_bytes"]
            and shape["evict_bytes"] >= 8 * meta["l2_bytes"],
            "Pool/eviction buffer smaller than 8x L2",
        )
    checks = [r for r in records if r["type"] == "correctness"]
    key = lambda record: case_key(record, blocks)
    require(
        collections.Counter(map(key, checks)) == collections.Counter(matrix),
        "Correctness matrix",
    )
    for record in checks:
        require(
            record["success"] is True and exact(record["rotations_checked"], [0, 1]),
            "Correctness check did not pass",
        )
    excluded = [r for r in records if r["type"] == "excluded_correctness"]
    excluded_matrix = (
        {("h2d", layout, n, "copy1d_batch") for layout in shapes for n in blocks}
        if exclude
        else set()
    )
    require(
        collections.Counter(map(key, excluded)) == collections.Counter(excluded_matrix),
        "Exclusion matrix",
    )
    for record in excluded:
        require(
            bool(record.get("reason")) and "success" not in record,
            "Exclusion cannot claim success",
        )
    mixed = [r for r in records if r["type"] == "mixed_crc_selftest"]
    require(
        exact(
            mixed,
            [
                {
                    "type": "mixed_crc_selftest",
                    "variant": "integrated_crc",
                    "success": True,
                }
            ],
        ),
        "Mixed CRC self-test failed",
    )
    corruption = [r for r in records if r["type"] == "crc_corruption"]
    require(
        collections.Counter((r["layout"], r["blocks"]) for r in corruption)
        == collections.Counter(itertools.product(shapes, (1, 8, 16, 32))),
        "Corruption matrix",
    )
    for record in corruption:
        case_key(record)
        require(
            record["variant"] == "integrated_crc"
            and record["direction"] == "h2d"
            and exact(record["cases"], 6)
            and exact(record["corruption_cases"], 6),
            "Invalid corruption check",
        )
        for field in (
            "success",
            "all_targets_unchanged",
            "recovery_success",
            "good_bad_good",
            "entire_pool_guards",
        ):
            require(record[field] is True, f"CRC check failed: {field}")

    pairs = collections.defaultdict(dict)
    for record in records:
        if record["type"] != "sample":
            continue
        require(key(record) in matrix, "Unexpected or excluded sample")
        require(
            type(record["repeat"]) is int and record["repeat"] == repeat,
            "Sample from wrong repeat",
        )
        require(
            record["timing"] == "wall" and record["regime"] == "cold",
            "Wrong timing method",
        )
        require(
            type(record["round"]) is int and 0 <= record["round"] < rounds,
            "Invalid round",
        )
        require(
            type(record["position"]) is int
            and record["position"]
            in range(len(active_variants(record["direction"], exclude))),
            "Invalid execution position",
        )
        require(
            type(record["source_plan"]) is int
            and 0
            <= record["source_plan"]
            < math.ceil(
                (layouts[record["layout"]]["pool_blocks"] - 1) / record["blocks"]
            ),
            "Invalid source rotation",
        )
        require(
            type(record["us"]) in (float, int)
            and math.isfinite(record["us"])
            and record["us"] > 0,
            "Invalid duration",
        )
        pair = (
            record["direction"],
            record["layout"],
            record["blocks"],
            record["round"],
        )
        require(record["variant"] not in pairs[pair], "Duplicate sample")
        pairs[pair][record["variant"]] = record
    require(
        set(pairs) == set(itertools.product(DIRECTIONS, shapes, blocks, range(rounds))),
        "Missing sample round",
    )
    for pair, values in pairs.items():
        require(
            set(values) == set(active_variants(pair[0], exclude)),
            "Incomplete paired round",
        )
        require(
            {r["position"] for r in values.values()} == set(range(len(values))),
            "Duplicate execution position",
        )
        require(
            len({r["source_plan"] for r in values.values()}) == 1,
            "Unpaired source blocks",
        )
    return meta, pairs


def summarize(values):
    values = sorted(values)
    return {
        "count": len(values),
        "p50_us": statistics.median(values),
        "p95_us": values[math.ceil(0.95 * len(values)) - 1],
        "min_us": values[0],
        "max_us": values[-1],
    }


def analyze(paths, settings):
    require(len(paths) == 2, "Exactly two process repeats required")
    inputs, cases, warnings = [], collections.defaultdict(list), []
    shapes = expected_layout(
        settings["model"],
        settings["logical_tokens_per_block"],
        settings["kernel_tokens_per_block"],
    )
    for repeat, path in zip((80, 81), paths):
        meta, pairs = read_run(path, settings, repeat)
        inputs.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "metadata": meta,
            }
        )
        for (direction, layout, n, round_id), values in pairs.items():
            cases[direction, layout, n].append((repeat, round_id, values))
    for key in (
        "gpu_uuid",
        "gpu",
        "sm",
        "runtime",
        "driver",
        "l2_bytes",
        "source_commit",
        "base_commit",
        "binary_sha256",
        "shape_source",
        "staged_no_crc_source",
        "model",
        "logical_tokens_per_block",
        "kernel_tokens_per_block",
        "block_counts",
        "tp_size",
        "cp_size",
        "cp_mode",
    ):
        require(
            inputs[0]["metadata"][key] == inputs[1]["metadata"][key],
            f"Repeat mismatch: {key}",
        )
    output = []
    for (direction, layout, n), rows in sorted(cases.items()):
        case = {
            "direction": direction,
            "layout": layout,
            "blocks": n,
            "payload_bytes": shapes[layout]["payload_bytes"],
            "variants": {},
            "comparisons": {},
        }
        for variant in active_variants(direction, settings["exclude_1d_h2d"]):
            stats = summarize([values[variant]["us"] for _, _, values in rows])
            stats["by_repeat"] = {}
            for repeat in (80, 81):
                selected = [
                    (round_id, values)
                    for rep, round_id, values in rows
                    if rep == repeat
                ]
                stats["by_repeat"][repeat] = summarize(
                    [values[variant]["us"] for _, values in selected]
                )
                if settings["iterations"] >= 4:
                    quarters = [
                        statistics.median(
                            values[variant]["us"]
                            for round_id, values in selected
                            if round_id * 4 // settings["iterations"] == quarter
                        )
                        for quarter in range(4)
                    ]
                    stats["by_repeat"][repeat]["quarter_p50_us"] = quarters
                    if max(quarters) / min(quarters) > 1.10:
                        warnings.append(
                            {
                                "direction": direction,
                                "layout": layout,
                                "blocks": n,
                                "variant": variant,
                                "repeat": repeat,
                                "reason": "quarter p50 drift exceeds 10%",
                            }
                        )
            stats["by_position"] = {}
            for position in range(
                len(active_variants(direction, settings["exclude_1d_h2d"]))
            ):
                values = [
                    v[variant]["us"]
                    for _, _, v in rows
                    if v[variant]["position"] == position
                ]
                stats["by_position"][position] = (
                    summarize(values) if values else {"count": 0}
                )
            for label, groups in (
                ("repeat", stats["by_repeat"]),
                ("position", stats["by_position"]),
            ):
                medians = [
                    s["p50_us"]
                    for s in groups.values()
                    if s["count"] >= (10 if label == "position" else 1)
                ]
                if medians and max(medians) / min(medians) > 1.10:
                    warnings.append(
                        {
                            "direction": direction,
                            "layout": layout,
                            "blocks": n,
                            "variant": variant,
                            "reason": label + " p50 difference exceeds 10%",
                        }
                    )
            stats["payload_GBps_at_p50"] = (
                n * case["payload_bytes"] / stats["p50_us"] / 1000
            )
            case["variants"][variant] = stats
        for baseline in case["variants"]:
            if baseline == "integrated_crc":
                continue
            deltas = [v["integrated_crc"]["us"] - v[baseline]["us"] for _, _, v in rows]
            ratios = [
                100 * (v["integrated_crc"]["us"] / v[baseline]["us"] - 1)
                for _, _, v in rows
            ]
            case["comparisons"][baseline] = {
                "paired_median_delta_us": statistics.median(deltas),
                "paired_median_delta_percent": statistics.median(ratios),
            }
        output.append(case)
    return {
        "valid": True,
        "settings": settings,
        "inputs": inputs,
        "sample_count": sum(len(v) for rows in cases.values() for _, _, v in rows),
        "paired_round_count": sum(map(len, cases.values())),
        "cases": output,
        "warnings": warnings,
        "notes": [
            "BS is local backing count per GPU, not request batch or an eight-rank sum.",
            "Physical tiles come from main CacheConfigCreator/DeviceBlockPoolConfigHelper; sentinel block 0 is never selected.",
            "staged_no_crc uses the rebased shared CopyTileKernel; it is not a separate pristine-main build.",
            "All samples retained. p50 is median; p95 is nearest rank.",
            "Negative paired CRC delta means CRC is faster than the named baseline.",
            "8x L2 eviction is outside timing; no hardware cache-miss guarantee.",
            "staged_no_crc is the rebased production strategy and includes CPU packing; gather_control is a separate no-CRC diagnostic control.",
            "Explicit exclusions are unavailable, not passing tests or zero latency.",
        ],
    }


def write_results(result, output):
    output = Path(output)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    meta = result["inputs"][0]["metadata"]
    lines = [
        f"Source: `{meta['source_commit']}`; main base: `{meta['base_commit']}`.",
        f"GPU: {meta['gpu']} (SM{meta['sm']}); CUDA runtime: {meta['runtime']}.",
        f"DeepSeek V4 {meta['model'].title()} FP8, TP{meta['tp_size']}/CP{meta['cp_size']}; "
        f"logical tokens/block={meta['logical_tokens_per_block']}, "
        f"kernel tokens/block={meta['kernel_tokens_per_block']}, BS={','.join(map(str, meta['block_counts']))}, "
        "gen_num_per_cycle=0.",
        "",
        "Complete synchronous call latency, p50 / p95 (us).",
        "",
        "| Direction | Layout | BS | CRC | 1D batch | 3D batch | staged without CRC | gather control, no CRC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    with (output / "latency.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "direction",
                "layout",
                "bs",
                "variant",
                "status",
                "samples",
                "p50_us",
                "p95_us",
                "payload_GBps_at_p50",
            ]
        )
        for case in result["cases"]:
            cells = []
            for variant in VARIANTS:
                stats = case["variants"].get(variant)
                cells.append(
                    f"{stats['p50_us']:.3f} / {stats['p95_us']:.3f}"
                    if stats
                    else "N/A (explicitly excluded)"
                )
                writer.writerow(
                    [
                        case["direction"],
                        case["layout"],
                        case["blocks"],
                        variant,
                        "measured" if stats else "explicitly_excluded",
                        *(
                            [
                                stats[k]
                                for k in (
                                    "count",
                                    "p50_us",
                                    "p95_us",
                                    "payload_GBps_at_p50",
                                )
                            ]
                            if stats
                            else [0, "", "", ""]
                        ),
                    ]
                )
            lines.append(
                "| "
                + " | ".join(
                    [case["direction"], case["layout"], str(case["blocks"]), *cells]
                )
                + " |"
            )
    lines.extend(
        ["", *result["notes"], "", f"Drift warnings: {len(result['warnings'])}"]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
