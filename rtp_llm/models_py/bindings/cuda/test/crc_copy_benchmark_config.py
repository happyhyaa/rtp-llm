"""Shared model and cache-span configuration for the CRC copy benchmark."""

from dataclasses import dataclass


MAX_BACKINGS = 32
MIN_COMPRESSED_KERNEL_TOKENS = 128
COMPRESSED_KERNEL_ALIGNMENT = 128


@dataclass(frozen=True)
class ModelProfile:
    name: str
    num_layers: int
    hidden_size: int
    head_num: int
    indexer_topk: int
    o_groups: int
    ratios: tuple
    tp_size: int
    cp_size: int
    kv_cache_sharded: bool
    cp_mode: str
    default_logical_tokens_per_block: int
    default_kernel_tokens_per_block: int


def _ratios(name):
    if name == "pro":
        return tuple([128, 128] + [4 if layer % 2 == 0 else 128 for layer in range(2, 61)])
    if name == "flash":
        return tuple([0, 0] + [4 if layer % 2 == 0 else 128 for layer in range(2, 43)])
    raise ValueError(f"Unknown DSV4 model: {name!r}; expected 'pro' or 'flash'")


def model_profile(name):
    if name == "pro":
        return ModelProfile(
            name="pro",
            num_layers=61,
            hidden_size=7168,
            head_num=128,
            indexer_topk=1024,
            o_groups=16,
            ratios=_ratios("pro"),
            tp_size=8,
            cp_size=8,
            kv_cache_sharded=True,
            cp_mode="CP_RR",
            default_logical_tokens_per_block=128,
            default_kernel_tokens_per_block=128,
        )
    if name == "flash":
        return ModelProfile(
            name="flash",
            num_layers=43,
            hidden_size=4096,
            head_num=64,
            indexer_topk=512,
            o_groups=8,
            ratios=_ratios("flash"),
            tp_size=1,
            cp_size=1,
            kv_cache_sharded=False,
            cp_mode="NONE",
            default_logical_tokens_per_block=1024,
            default_kernel_tokens_per_block=128,
        )
    raise ValueError(f"Unknown DSV4 model: {name!r}; expected 'pro' or 'flash'")


def parse_block_counts(value):
    if isinstance(value, str):
        parts = value.split(",")
        if not parts or any(not part.strip().isdigit() for part in parts):
            raise ValueError("BS list must be comma-separated positive integers")
        value = tuple(int(part.strip()) for part in parts)
    else:
        value = tuple(value)
    return value


def resolve_benchmark_config(
    model,
    logical_tokens_per_block=None,
    kernel_tokens_per_block=None,
    block_counts=None,
):
    profile = model_profile(model)
    if logical_tokens_per_block is None:
        logical_tokens_per_block = profile.default_logical_tokens_per_block
    if kernel_tokens_per_block is None:
        kernel_tokens_per_block = profile.default_kernel_tokens_per_block
    if block_counts is None:
        block_counts = tuple(range(1, MAX_BACKINGS + 1))
    block_counts = parse_block_counts(block_counts)
    for value, label in (
        (logical_tokens_per_block, "logical tokens/block"),
        (kernel_tokens_per_block, "kernel tokens/block"),
    ):
        if type(value) is not int or value <= 0 or value > 2**32 - 1:
            raise ValueError(f"{label} must be a positive uint32 integer")

    if logical_tokens_per_block < kernel_tokens_per_block:
        raise ValueError("logical tokens/block must be >= kernel tokens/block")
    if logical_tokens_per_block % kernel_tokens_per_block:
        raise ValueError("logical tokens/block must be divisible by kernel tokens/block")
    if (
        kernel_tokens_per_block < MIN_COMPRESSED_KERNEL_TOKENS
        or kernel_tokens_per_block % COMPRESSED_KERNEL_ALIGNMENT
    ):
        raise ValueError(
            "compressed DSV4 kernel tokens/block must be a positive multiple of 128"
        )
    if any(
        ratio > 1 and kernel_tokens_per_block % ratio
        for ratio in profile.ratios
    ):
        raise ValueError("kernel tokens/block must be divisible by every model compression ratio")

    counts = block_counts
    if not counts:
        raise ValueError("BS list must not be empty")
    if any(type(value) is not int or value < 1 or value > MAX_BACKINGS for value in counts):
        raise ValueError(f"each BS must be an integer in [1, {MAX_BACKINGS}]")
    if len(set(counts)) != len(counts):
        raise ValueError("BS list must not contain duplicates")

    return {
        "model": profile.name,
        "logical_tokens_per_block": logical_tokens_per_block,
        "kernel_tokens_per_block": kernel_tokens_per_block,
        "block_counts": counts,
        "profile": profile,
    }
