"""任务 11（选做）：DistilBERT 微调。实现在 hf_trainer.py。

DistilBERT 是 BERT 的知识蒸馏版本：12 层砍到 6 层，参数从 1.1 亿降到 6,600 万，
推理快约 60%，在多数分类任务上只掉 1 个点左右。是三个预训练模型里性价比最高的，
建议先跑这个。

    python experiments/finetune/distilbert.py
    python experiments/finetune/distilbert.py --epochs 2 --batch-size 32
    python experiments/finetune/distilbert.py --predict "A quietly devastating film."
"""

import sys
from pathlib import Path

# 允许 `python experiments/.../x.py` 直接运行（不必写成 python -m）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import hf_trainer

NAME = "distilbert"

if __name__ == "__main__":
    parser = hf_trainer.build_parser(model_id="distilbert-base-uncased",
                                     batch_size=32, lr=5e-5, epochs=2)
    hf_trainer.run(NAME, parser.parse_args())
