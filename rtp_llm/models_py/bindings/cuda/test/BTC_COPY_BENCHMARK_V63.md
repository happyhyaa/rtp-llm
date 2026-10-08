# BTC interface benchmark v6.3

Method migrated from `alibaba/rtp-llm:codex/dsv4-batched-crc` at
`eb70584558fe102e1e0eae9c073f939459cdf522` (CrcCopyBenchmark and its runner/stats).
Production baseline is `release/btc_1.0@3d7a58e7a2ff2c311a7a780f29687e20c04a8297`
plus CUDA 3D integration `13eb990a402defd04cb3dd3e23bee16b6c0daa99`.

Only the directly linked production CUDA 1D batch and staged SM strategies are
measured. A non-DONE result is an error, never an implicit generic fallback.
3D remains in the production tree but is not part of this measurement.

The reference method is retained: H2D/D2H, Full/SWA, every BS 1..32, two separate
process repeats, 30 warmup and 100 measured paired rounds per point, randomized
strategy order, identical rotating blocks per pair, independent 8x L2 eviction
before each call outside timing, CPU descriptor priming, and synchronous wall
latency. Native plans are built outside timing; strategy-internal construction,
packing, locking, submission and synchronization remain inside timing.
CPU payload oracle, entire device pool guards and host guards run at two block
rotations for every BS before timing. P50/P95 and drift checks retain the original
algorithm. Raw data and binary/source provenance are retained.

Intentional adaptations: Flash FP8 TP1/CP1 replaces Pro TP8/CP8; both block sizes
are 128. Pool geometry is derived from the current CacheConfigCreator and
DeviceBlockPoolConfigHelper. CRC, diagnostic gather and experimental 3D cases are
removed. Host allocation stride preserves the reference alignment/padding rule,
but only payload bytes are copied and all padding is checked as guard bytes.
Production shared-stream execution contexts are supplied to the current API.
