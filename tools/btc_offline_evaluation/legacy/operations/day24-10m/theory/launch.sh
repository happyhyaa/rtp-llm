#!/bin/bash
set -eu
cd "$(dirname "$0")"
base=/data0/REPLAY_USER/hit-rate-evaluation
grid=$base/cp4-gridsearch-20261103-v1/cp4_gridsearch
up=$base/finite-device-20261030-v1/upstream
. /opt/rh/gcc-toolset-12/enable
g++ -std=c++17 -O3 -pthread -I "$grid" -I "$up" hold_worker.cc "$up"/kv_cache_manager/common/cache/{cache,lru_cache,sharded_cache}.cc -o hold_worker
exec ./hold_worker "$base/sgp04-cp4-b256-20261102-v3/input1024" state
