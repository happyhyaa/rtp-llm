---
name: btc-offline-evaluation
description: Inspect, materialize, build, or operate this repository's archived DSv4.1 BTC CPU metadata hit-rate replay. Use for native BTC prefill match/insert/eviction replay, trace preparation, capacity comparisons, and interpreting the archived results.
---

# BTC offline evaluation archive

Resolve the archive root as two directories above this skill folder. Read `README.md`, `archive-manifest.json`, and `docs/known-limitations.md` before selecting a runnable entry point. Read `docs/legacy-design.md` for binary input, CP, SWA tail and eviction semantics. `docs/next-design.md` contains pending design rather than completed behavior.

## Materialize and verify

Run `python3 <archive>/scripts/check_archive.py` to verify source digests. The baseline commit in the manifest is authoritative. In an isolated worktree at that baseline, copy this archive and apply `legacy/source.patch` with `git apply --check` followed by `git apply`. Never apply it blindly to another revision or overwrite uncommitted source.

The historically exercised noon-version target is `//rtp_llm/cpp/cache/block_tree_cache/benchmark:resident_metadata_replay`. Existing JSONL fixture helpers retain earlier-interface assumptions. Build in a matching Linux environment. Distinguish compiling CUDA dependencies from executing GPU work. Actual KV payload allocation, physical copies and GPU execution are excluded from the metadata replay.

## Inputs and experiments

Preserve request order and original lengths. Supply complete base-granularity prefix keys; native CPSlotMapper performs CP canonical selection. Apply a prefix transform only when the input hash contract requires it. Use source/output digests and the input manifest to identify a cohort.

Record effective capacities by group and tier, base block tokens, CP mapping, SWA window, reuse_cache_tail_blocks, watermarks and eviction policies. Historical last_n groups use Device SWA=1000*n and change FULL/SWA quotas. Full-retention groups use 1:1 counts. Counts are usable blocks per rank, with reserved zero slots additional.

Historical operation scripts contain deployment placeholders and explicit request limits. User authorization governs starting, repeating or resuming experiments. A request to inspect or package this archive does not authorize a benchmark. Distinguish cold-segment experiments from hot continuation.

## Report and resume

Compare identical request ranges and denominators. hit-tokens.u32 contains per-request hit lengths; nested match timing is included in malloc timing. Attribute findings only to observed evidence. Keep simulator-only, native BTC metadata and online inference measurements labeled.

The old held process can continue only while its in-memory state survives. JSON summaries are not restorable checkpoints. Do not claim restartable snapshots, deterministic repeatability, F/J metrics, or historical branchpoint tracking exist until implemented and verified.
