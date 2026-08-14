"""阶段三：DeBERTa + P-Tuning 微调 IMDB 情感分类。

P-Tuning（Liu et al. 2021）是三种 prompt 类方法里最轻的一种：只在**输入层**
（不是每一层）前面拼 `num_virtual_tokens` 个虚拟 token 的 embedding。

关键设计是这些 embedding **不直接优化**，而是由一个小的 prompt encoder
（peft 默认用 MLP，也可选 LSTM，`encoder_hidden_size=128`）生成：

    [p_1 ... p_k] = PromptEncoder(可训练的 k 个索引)

为什么要多这一层间接。直接优化 k 个独立的 embedding 向量时，它们之间没有任何
耦合，优化面很崎岖，小模型上经常收敛不了或者结果方差极大；套一个共享的
encoder 后参数被绑在一起，训练稳定得多。训练结束后 encoder 可以丢掉，
只留生成好的 k 个向量。

和 Prefix-Tuning 的对比正好构成一组消融：都是虚拟 token，
**只插输入层（P-Tuning）** vs **每层都插（Prefix）**，看深度带来多少收益。
和 LoRA 的对比则是另一维：**改输入** vs **改权重**。

由于只动输入 embedding，它不需要 `past_key_values`，在 DeBERTa 这种纯 encoder 上
可以正常跑——这也是 Prefix 若失败时的对照组。

    python experiments/peft/ptuning.py --probe-steps 20
"""

import sys
from pathlib import Path

# 允许 `python experiments/.../x.py` 直接运行（不必写成 python -m）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import peft_trainer

if __name__ == "__main__":
    args = peft_trainer.build_parser(method="ptuning").parse_args()
    peft_trainer.run("deberta_ptuning", args)
