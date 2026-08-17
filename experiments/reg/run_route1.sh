#!/usr/bin/env bash
# 路线①（改 forward）的三组实验：BERT-base 全量微调
#   baseline —— imdb_bert_scl.py --alpha 0，SCL 权重为 0 就退化成纯交叉熵，
#               不用再单独写一个脚本
#   rdrop    —— imdb_bert_rdrop.py
#   scl      —— imdb_bert_scl.py
#
# max_length / batch / epochs 跟 run_all.sh 保持一致，方便两条路线互相参照。
# 注意：这里是全量微调（不挂 LoRA），lr 用 2e-5，不能用 LoRA 的 2e-4。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

COMMON="--max_length 256 --batch_size 16 --grad_accum 1 --epochs 2 --lr 2e-5"

echo "===== $(date '+%F %T') start route1 baseline"
python imdb_bert_scl.py $COMMON --alpha 0 \
  --submission ../../submissions/17_rdrop_scl/bert-base_full_none.csv > logs/run_route1_none.log 2>&1
echo "===== $(date '+%F %T') done route1 baseline"

echo "===== $(date '+%F %T') start route1 rdrop"
python imdb_bert_rdrop.py $COMMON --alpha 1.0 \
  --submission ../../submissions/17_rdrop_scl/bert-base_full_rdrop.csv > logs/run_route1_rdrop.log 2>&1
echo "===== $(date '+%F %T') done route1 rdrop"

echo "===== $(date '+%F %T') start route1 scl"
python imdb_bert_scl.py $COMMON --alpha 0.2 --temperature 0.3 \
  --submission ../../submissions/17_rdrop_scl/bert-base_full_scl.csv > logs/run_route1_scl.log 2>&1
echo "===== $(date '+%F %T') done route1 scl"

echo "ROUTE1 ALL DONE"
