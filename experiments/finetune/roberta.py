"""任务 11（选做）：RoBERTa-base 微调。实现在 hf_trainer.py。

RoBERTa 的结构和 BERT 几乎一样，赢在预训练配方：去掉 NSP 任务、改用动态遮盖、
语料从 16 GB 加到 160 GB、批大小和训练步数都大幅提高。
「同样的架构，更充分的预训练」——通常比 BERT-base 高 1~2 个点。

学习率要比 BERT 更小（默认 1e-5）：RoBERTa 微调时对大学习率更敏感，
5e-5 常见的表现是 loss 卡在 0.69、准确率停在 0.5（塌成全预测同一类）。

    python experiments/finetune/roberta.py
    python experiments/finetune/roberta.py --epochs 2 --lr 1e-5
    python experiments/finetune/roberta.py --predict "Overlong and self-indulgent."
"""

import sys
from pathlib import Path

# 允许 `python experiments/.../x.py` 直接运行（不必写成 python -m）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import hf_trainer

NAME = "roberta"

if __name__ == "__main__":
    parser = hf_trainer.build_parser(model_id="roberta-base",
                                     batch_size=16, lr=1e-5, epochs=2)
    hf_trainer.run(NAME, parser.parse_args())
