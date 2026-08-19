#!/usr/bin/env bash
# 第一阶段：真实 batch_size=4 的三组对照。
#
# 三组之间只差 --method。其余全部走 run_experiment.py 的默认值，
# 也就是 0.9633 那次 DeBERTa-v3-large + LoRA 的配置：
#   max_length 384 / lr 1e-4 / epochs 2 / seed 42 / r16 alpha32 dropout0.05
#   grad_accum 自动 = 32/4 = 8 → effective batch 32（与 0.9633 一致）
#   λ=0.2 τ=0.3 queue_size=4096 m=0.999
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs result

BS="${BS:-4}"
SEED="${SEED:-42}"
for method in baseline scl scl_moco; do
  tag="deberta_lora_${method}_bs${BS}_seed${SEED}"
  echo "===== $(date '+%F %T') start ${tag}"
  python run_experiment.py --method "$method" --batch_size "$BS" --seed "$SEED" \
    || { echo "!!!!! ${tag} 失败，见 logs/${tag}.log"; exit 1; }
  echo "===== $(date '+%F %T') done ${tag}"
done
echo "STAGE bs=${BS} ALL DONE"
python collect_results.py || true
