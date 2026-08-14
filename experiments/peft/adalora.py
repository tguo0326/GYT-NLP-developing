"""阶段三：DeBERTa + AdaLoRA 微调 IMDB 情感分类。

AdaLoRA（Adaptive Budget Allocation for PEFT, Zhang et al. 2023）针对 LoRA 的一个
硬伤：所有层共用同一个秩 r。但实际上不同层需要的容量差别很大——底层学通用特征、
顶层学任务特征，把 r 平摊过去等于浪费预算。

做法是把增量写成 SVD 的形式并让秩可变：

    ΔW = P · Λ · Q            Λ 是对角的奇异值

训练时按每个奇异值的**重要性分数**（梯度×权重的滑动平均）排序，
定期把不重要的三元组（p_i, λ_i, q_i）置零，把释放出来的预算让给更需要的层。
调度分三段：`tinit` 步 warmup 不裁剪 → 中间按 `deltaT` 逐步裁到 `target_r`
→ 最后 `tfinal` 步固定下来继续训练。

所以本脚本从 `init_r = 2r` 起步，最终平均裁到 `target_r = r`：
和 LoRA 的最终参数预算相当，比的是「同样的预算，会不会分配」。
代价是训练中多维护一份重要性统计，比 LoRA 略慢。

    python experiments/peft/adalora.py
    python experiments/peft/adalora.py --probe-steps 20
"""

import sys
from pathlib import Path

# 允许 `python experiments/.../x.py` 直接运行（不必写成 python -m）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import peft_trainer

if __name__ == "__main__":
    args = peft_trainer.build_parser(method="adalora").parse_args()
    peft_trainer.run("deberta_adalora", args)
