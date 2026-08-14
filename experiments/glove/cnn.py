"""任务 7：TextCNN 情感分类（GloVe 词向量 + 一维卷积）。

CNN 怎么提取文本局部特征：Embedding 之后每条评论是 (512, 300) 的矩阵，
`Conv1d(300, num_filter, filter_size)` 在时间轴上滑窗——一个 filter_size=3 的卷积核
每次看连续 3 个词，输出一个标量。128 个卷积核就是 128 个「三元词组探测器」，
有的会对 `not very good` 高响应，有的对 `one of the best` 高响应。
随后 `max_pool1d` 沿整条序列取最大值：**只关心这个特征在评论里出现过没有、
最强出现在哪，不关心出现在第几个词**。这就是 CNN 做文本分类的核心假设——
情感由若干局部短语决定，绝对位置不重要。

相比原始版本（docs/original_code/imdb_cnn.py）的修改：

* pickle 文件名 `imdb_demo_glove.pickle3` → 实际产出的 `imdb_glove.pickle3`；
* 训练循环、设备选择、种子、日志、最佳模型保存全部移交 common.py；
* 单一 filter_size=3 扩展成多尺度 (3, 4, 5) 并行卷积——单一尺度只能看三元词组，
  多尺度是 TextCNN 原论文的做法，代价只有几十万参数；
* 加 Dropout：原代码没有任何正则化，即便冻结 Embedding 也是两三个 epoch 就过拟合；
* 优化器 SGD(lr=0.8) → Adam(lr=1e-3)。lr=0.8 配 SGD 在这个网络上 loss 会震荡不收敛。

用法：

    python experiments/glove/cnn.py
    python experiments/glove/cnn.py --epochs 8 --filter-sizes 3 4 5
    python experiments/glove/cnn.py --predict "This film was a masterpiece."
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许 `python experiments/.../x.py` 直接运行（不必写成 python -m）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn, optim
from torch.nn import functional as F

from core import common

NAME = "cnn"


class SentimentNet(nn.Module):
    def __init__(self, weight: torch.Tensor, num_filter: int = 128,
                 filter_sizes: tuple[int, ...] = (3, 4, 5), labels: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        embed_size = weight.shape[1]
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=0)
        # 每个尺度一条卷积分支。padding=size//2 保证短评论也能卷。
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_size, num_filter, size, padding=size // 2)
            for size in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(num_filter * len(filter_sizes), labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (batch, seq) → (batch, seq, embed) → (batch, embed, seq)：Conv1d 要求通道在中间
        embeddings = self.embedding(inputs).permute(0, 2, 1)
        pooled = []
        for conv in self.convs:
            features = F.relu(conv(embeddings))            # (batch, num_filter, seq')
            # 沿整条序列取最大值：特征「是否出现」比「出现在哪」更重要
            pooled.append(F.max_pool1d(features, features.shape[2]).squeeze(2))
        return self.decoder(self.dropout(torch.cat(pooled, dim=1)))


def main() -> None:
    parser = common.build_parser(epochs=10, batch_size=64, lr=1e-3)
    parser.add_argument("--num-filter", type=int, default=128)
    parser.add_argument("--filter-sizes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--dropout", type=float, default=0.5)
    args = parser.parse_args()

    common.run(
        NAME, args,
        lambda bundle: SentimentNet(bundle.weight, num_filter=args.num_filter,
                                    filter_sizes=tuple(args.filter_sizes),
                                    dropout=args.dropout),
        lambda model: optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr),
    )


if __name__ == "__main__":
    main()
