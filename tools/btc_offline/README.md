# BTC offline prefill replay

Build the replay program in a DSv4.1 build environment:

```sh
bazel build --config=cuda13 //tools/btc_offline:replay
```

The target uses the BTC implementation in this checkout. The metadata-only build
configuration excludes the model execution library. The current BTC library still
has native transfer dependencies. At replay time an injected submitter validates
source and target block IDs, reports buffer safety, and completes the native transfer
callback without copying KV data. The pools keep allocation and reference metadata
but allocate no KV payload. Model execution and decode are outside this prefill driver.

The source marker in `btc-offline-backend.json` identifies this target and its
three validation binaries for the replay skill. Use the corresponding RTP-LLM
runtime libraries and run without GPU devices. Validate pool accounting, native
watermarks, allocation failure, checkpoint restore and smoke replay before comparing
hit rates. Checkpoints and measurements belong to the binary and configuration
that created them.
