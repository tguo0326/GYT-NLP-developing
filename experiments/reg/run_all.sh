#!/usr/bin/env bash
# 四组对比实验：LoRA baseline / +R-Drop / +SCL / +两者
# 骨干固定 ModernBERT-large，除 --reg 外所有超参完全一致，保证可比。
#
# 配置理由（单卡 Tesla T4 15G，fp16）：
#   max_length 256  —— 512 时训练吞吐只有 5.3 samples/s，四组要 ~17h；256 约 2 倍速
#   batch_size 16   —— SCL 需要 batch 内有同类样本；二分类下 16 足够
#   epochs 2        —— 20000 条训练集，LoRA lr 2e-4，2 epoch 已收敛
set -euo pipefail
cd "$(dirname "$0")"

# PYTHON=../.venv-unsloth/bin/python TAG_SUFFIX=_unsloth ./run_all.sh  跑 unsloth 后端
# Triton 需要系统 gcc + 系统 python 头文件，conda 的交叉编译器看不到它们
PYTHON="${PYTHON:-python}"
TAG_SUFFIX="${TAG_SUFFIX:-}"
export CC="${CC:-/usr/bin/gcc}" CXX="${CXX:-/usr/bin/g++}"
export CPATH="${CPATH:-/usr/include/x86_64-linux-gnu}"

mkdir -p logs
for reg in none rdrop scl both; do
  echo "===== $(date '+%F %T') start reg=$reg"
  "$PYTHON" imdb_unsloth_lora.py \
    --tag_suffix "$TAG_SUFFIX" \
    --model_name answerdotai/ModernBERT-large \
    --reg "$reg" \
    --max_length 256 \
    --batch_size 16 \
    --grad_accum 1 \
    --epochs 2 \
    --lr 2e-4 \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --rdrop_alpha 1.0 --scl_alpha 0.2 --scl_temperature 0.3 \
    > "logs/run_${reg}${TAG_SUFFIX}.log" 2>&1
  echo "===== $(date '+%F %T') done reg=$reg"
done
echo "ALL DONE"
