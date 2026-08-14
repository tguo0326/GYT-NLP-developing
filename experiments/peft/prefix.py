"""阶段三：DeBERTa + Prefix-Tuning 微调 IMDB 情感分类。

Prefix-Tuning（Li & Liang, 2021）比 LoRA 更早，思路也不同：一个字都不改模型权重，
而是在**每一层**自注意力的 K 和 V 前面拼上 `num_virtual_tokens` 个可训练向量。

    K' = [P_k ; K]        V' = [P_v ; V]

真实 token 在算注意力时能"看到"这些前缀，相当于给每一层都注入了一段
连续的、可学习的"任务指令"——是 discrete prompt（写一段自然语言提示词）的
连续化版本，不受词表限制，梯度能直接优化。

两个和 LoRA 的关键差别：
1. 前缀占掉了注意力的位置预算——20 个虚拟 token 会挤掉同样长度的有效上下文；
2. 推理时无法合并回原权重，每次前向都要多算这段前缀。

**为什么这个脚本的默认底座不是 DeBERTa**（实测结论，不是代码 bug）：
peft 通过 `past_key_values` 把前缀塞进每一层，这要求底座实现 KV cache。
DeBERTa 是纯 encoder，`DebertaV2Model` 根本没有这个入口，实跑直接抛

    ValueError: Model does not support past key values which are
                required for prefix tuning.

而且 peft 还额外禁止 Prefix-Tuning 与 gradient checkpointing 同时开
（`PREFIX_TUNING does not work with gradient checkpointing.`）——
道理一样：checkpoint 段反向重算时那份 KV cache 已经不在图里了。

所以这里换成 `roberta-base`（支持 past_key_values，已验证可跑），
在对比表里作为**带脚注的补充行**，不与另外三种方法直接比准确率——
底座不同，比了也不公平。这本身是 PEFT 的一个实用结论：
**方法的可用性受底座架构限制，LoRA 系几乎通吃，prompt 系挑架构。**
详见 docs/peft-lora.md。

    python experiments/peft/prefix.py                        # 默认 roberta-base
    python experiments/peft/prefix.py --model-id microsoft/deberta-v3-large  # 会报错，可复现上述结论
"""

import sys
from pathlib import Path

# 允许 `python experiments/.../x.py` 直接运行（不必写成 python -m）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import peft_trainer

if __name__ == "__main__":
    args = peft_trainer.build_parser(method="prefix",
                                     model_id="roberta-base").parse_args()
    peft_trainer.run("deberta_prefix", args)
