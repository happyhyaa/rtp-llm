set -eu
cd "$(dirname "$0")"
base=/data0/REPLAY_USER/hit-rate-evaluation
prepared=$base/last-n-20261029-v1/prepared
reference=$base/sgp04-cp4-b256-20261102-v3
up=$base/finite-device-20261030-v1/upstream
python3 preflight.py . "$prepared" "$reference/input1024" bailian-20261027-24h.json
g++ -std=c++17 -O3 -I"$reference/sgp04_cp4" -I"$up" convert.cc "$up/kv_cache_manager/optimizer/liteHit/request_preprocess.cc" -o convert
exec ./convert "$prepared" "$reference/input1024" .
