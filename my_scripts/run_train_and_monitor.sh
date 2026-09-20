#!/usr/bin/env bash
# run_train_and_monitor.sh
#
# 训练 + 并行采 GPU 利用率/显存。统计时长 = 训练时长。
# 用法: bash run_train_and_monitor.sh [gpu_id] [yaml_path]
#      默认 gpu_id=3, yaml=my_configs/qwen3_8b_train_qlora.yaml
# 产出(每次运行建一个时间戳子目录,位于 /mnt/data/dingshang/my_outputs/<时间戳>/):
#   train.log        训练完整 stdout
#   gpu_stats.csv    (elapsed_s,gpu_util_pct,gpu_mem_mb)
#   summary.txt      汇总统计
#   <yaml_name>.yaml 当时用的配置快照

set -u

GPU=${1:-3}
YAML=${2:-my_configs/qwen3_8b_train_qlora.yaml}
LLAFA=/mnt/data/dingshang/LLaMA-Factory
PY=/mnt/data/conda/envs/dslf/bin/python
CONDA=/mnt/data/conda/bin/conda
OUT_BASE=/mnt/data/dingshang/my_outputs
YAML_TAG=$(basename "$YAML" .yaml)
OUT_DIR="$OUT_BASE/$(date +%Y%m%d_%H%M%S)_${YAML_TAG}"
CSV="$OUT_DIR/gpu_stats.csv"
LOG="$OUT_DIR/train.log"

mkdir -p "$OUT_DIR"
cp "$YAML" "$OUT_DIR/"
cd "$LLAFA"

# 后台:每 1 秒采一次 GPU 3 的 util/mem,写到 CSV
"$PY" - "$GPU" "$CSV" <<'PY' &
import sys, subprocess, time
gpu, path = sys.argv[1], sys.argv[2]
open(path, 'w').write('elapsed_s,gpu_util_pct,gpu_mem_mb\n')
t0 = time.time()
while True:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
             '--format=csv,noheader,nounits', '-i', gpu],
            timeout=2).decode().strip()
        u, m = out.split(',')
    except Exception:
        u = m = '-1'
    with open(path, 'a') as f:
        f.write(f'{time.time()-t0:.2f},{u},{m}\n')
    time.sleep(1.0)
PY
STATS_PID=$!
trap "kill $STATS_PID 2>/dev/null" EXIT INT TERM

# 前台:跑训练,stdout 同时进终端和 log
echo "[INFO] running: llamafactory-cli train $YAML on GPU $GPU"
echo "[INFO] output dir: $OUT_DIR"
CUDA_VISIBLE_DEVICES=$GPU "$CONDA" run -n dslf \
    llamafactory-cli train "$YAML" 2>&1 | tee "$LOG"

# 训练结束 → 杀监控 → 等监控退出 → 打统计
kill $STATS_PID 2>/dev/null
wait $STATS_PID 2>/dev/null

"$PY" - "$CSV" "$OUT_DIR/summary.txt" <<'PY'
import csv, sys
csv_path, summary_path = sys.argv[1], sys.argv[2]
rows = [r for r in csv.reader(open(csv_path))][1:]
utils = [int(r[1]) for r in rows if r[1] != '-1']
mems = [int(r[2]) for r in rows if r[2] != '-1']
text = (
    f'samples : {len(rows)}\n'
    f'util %  : min={min(utils)}  max={max(utils)}  avg={sum(utils)/len(utils):.1f}\n'
    f'mem MB  : min={min(mems)}  max={max(mems)}  avg={sum(mems)/len(mems):.0f}'
)
print('\n=== GPU stats ===')
print(text)
with open(summary_path, 'w') as f:
    f.write(text + '\n')
PY