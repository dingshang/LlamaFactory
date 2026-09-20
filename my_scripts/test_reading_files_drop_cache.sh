#!/usr/bin/env bash
# test_reading_files_drop_cache.sh
#
# 两组对照:
#   1. 热 cache (vmtouch -t) + 不丢 cache   → 全 cache hit
#   2. 冷 cache (vmtouch -e) + 每行丢 cache → 全 disk 读
#
# 用法:
#   bash my_scripts/test_reading_files_drop_cache.sh [file]
#   默认 file=data/streaming_alpaca_zh_demo_512mb.jsonl

set -u

FILE=${1:-data/streaming_alpaca_zh_demo_512mb.jsonl}
FUSE_FILE=my_fuse_data/streaming_alpaca_zh_demo_512mb.jsonl
SCRIPT=my_scripts/file_reading_drop_cache.py
PY=${PY:-python3}

if [[ ! -f "$FILE" ]]; then
    echo "[error] file not found: $FILE" >&2
    exit 1
fi

echo "[1] 热 cache + not drop (全 cache hit)"
vmtouch -t "$FILE"
vmtouch "$FILE"
"$PY" "$SCRIPT" "$FILE" false
vmtouch "$FILE"
echo

echo "[2] 冷 cache + drop (读后建议内核丢弃 page cache)"
vmtouch -e "$FILE"
vmtouch "$FILE"
"$PY" "$SCRIPT" "$FILE" true
vmtouch "$FILE"

echo "[3] fuse data + no drop (全 cache hit)"
"$PY" "$SCRIPT" "$FUSE_FILE" false

echo "[4] brickcache pybind + no drop (全 cache hit)"
export PYTHONPATH=${PYTHONPATH:-}:/mnt/data/dingshang/BrickCache/python:/mnt/data/dingshang/BrickCache/build/python
"$PY" my_scripts/file_reading_drop_cache_brickcache.py "/data1/streaming_alpaca_zh_demo_512mb.jsonl" false
