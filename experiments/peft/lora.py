"""阶段三：DeBERTa + LoRA 微调 IMDB 情感分类。

LoRA（Low-Rank Adaptation of Large Language Models, Hu et al. 2021）的出发点是
一个经验观察：微调带来的权重更新量 ΔW 虽然形状和 W 一样大，但**内在秩很低**——
把一个 1024×1024 的更新压到秩 16 几乎不掉精度。于是干脆不去更新 W，
而是把 ΔW 显式写成两个瘦矩阵的乘积：

    h = Wx + (α/r) · B·A·x        A ∈ R^{r×d}（高斯初始化）, B ∈ R^{d×r}（零初始化）

B 初始化为零使训练开始时 BA=0，模型行为与原底座完全一致，不会一上来就被打乱。
r=16 时，一个 1024 维的投影只多 2×1024×16 ≈ 3.3 万个参数，而原矩阵有 105 万个。
本脚本只给注意力的 Q/V 投影挂 LoRA（论文的消融实验里 Q+V 的性价比最高）。

三个实际好处：
1. 可训练参数降到 0.x%，优化器状态跟着降两个数量级（Adam 每参数 2 份动量）；
2. 训完可以 `merge_and_unload()` 把 BA 加回 W，推理时零额外延迟——
   这是 LoRA 相比 Adapter（串联额外层）的关键优势；
3. 一个底座配多个 adapter，每个任务只存几 MB。

    python experiments/peft/lora.py                          # 正式训练
    python experiments/peft/lora.py --probe-steps 20         # 只测显存峰值
    python experiments/peft/lora.py --batch-size 4           # 显存不够就调小
"""

import sys
from pathlib import Path

# 允许 `python experiments/.../x.py` 直接运行（不必写成 python -m）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import peft_trainer

if __name__ == "__main__":
    args = peft_trainer.build_parser(method="lora").parse_args()
    peft_trainer.run("deberta_lora", args)
