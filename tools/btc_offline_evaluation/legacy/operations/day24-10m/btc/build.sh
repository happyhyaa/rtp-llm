#!/bin/bash
set -eu
cd /data0/REPLAY_USER/hit-rate-evaluation/btc-serial-20261103/source
export USER=REPLAY_USER USE_BAZEL_VERSION=6.4.0 CUDA_VISIBLE_DEVICES=
bazelisk --output_user_root=/data0/REPLAY_USER/hit-rate-evaluation/btc-serial-20261103/bazel-cache build --config=cuda13 --jobs=16 --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=8.9 --action_env=TF_CUDA_COMPUTE_CAPABILITIES=8.9 --host_action_env=TF_CUDA_COMPUTE_CAPABILITIES=8.9 //rtp_llm/cpp/cache/block_tree_cache/benchmark:resident_metadata_replay
