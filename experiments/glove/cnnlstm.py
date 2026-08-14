"""任务 10-A：CNN-LSTM 组合模型（GloVe 词向量）。

思路是让两种归纳偏置各干各擅长的事：

* **CNN 层当特征压缩器**：卷积核在词序列上滑窗，把连续几个词的组合抽成
  `num_filter` 维的短语特征；紧接着沿时间轴做 stride=2 的最大池化，
  序列长度 512 → 256。相当于把「词序列」改写成「短语序列」；
* **LSTM 层当顺序聚合器**：在这条更短、语义更粗的短语序列上建模顺序关系。
  序列短一半，LSTM 的时间步就少一半，训练更快，长依赖也更容易学。

相比原始版本（docs/original_code/imdb_cnnlstm.py）修了一个实质性的维度错误：

    # 原代码
    pooling = F.max_pool1d(convolution, kernel_size=pooling_size)  # (batch, 128, 256)
    states, hidden = self.encoder(pooling.permute([1, 0, 2]))       # (128, batch, 256)
    self.encoder = nn.LSTM(input_size=max_len // pooling_size, ...) # input_size=256

`permute([1, 0, 2])` 把 **卷积核维度（128）当成了时间步**、把 **时间步（256）当成了
特征维度**。也就是说 LSTM 在「128 个卷积核」这个无序集合上做顺序建模，
而真正的词序信息被塞进了特征向量里——组合模型的意义完全没了。
正确的形状是 `(batch, seq', num_filter)`，`input_size=num_filter`。

其余修改与 CNN/LSTM 两个脚本一致：加 dropout、SGD(lr=0.8) → Adam(lr=1e-3)、
训练循环移交 common.py。

用法：

    python experiments/glove/cnnlstm.py
    python experiments/glove/cnnlstm.py --predict "Visually stunning but emotionally hollow."
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

NAME = "cnnlstm"


class SentimentNet(nn.Module):
    def __init__(self, weight: torch.Tensor, num_filter: int = 128, filter_size: int = 3,
                 pooling_size: int = 2, num_hiddens: int = 64, num_layers: int = 2,
                 bidirectional: bool = True, labels: int = 2, dropout: float = 0.5):
        super().__init__()
        embed_size = weight.shape[1]
        self.pooling_size = pooling_size
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=0)

        self.conv1d = nn.Conv1d(embed_size, num_filter, filter_size, padding=filter_size // 2)
        # LSTM 吃的是卷积核维度，不是被池化后的序列长度——这是原代码搞反的地方
        self.encoder = nn.LSTM(
            input_size=num_filter, hidden_size=num_hiddens, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(num_hiddens * self.num_directions * 2, labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(inputs).permute(0, 2, 1)   # (batch, embed, seq)
        features = F.relu(self.conv1d(embeddings))             # (batch, filter, seq)
        pooled = F.max_pool1d(features, self.pooling_size)     # (batch, filter, seq/2)
        sequence = pooled.permute(0, 2, 1)                     # (batch, seq/2, filter)

        states, (hidden, _cell) = self.encoder(sequence)
        last = hidden[-self.num_directions:].permute(1, 0, 2).flatten(1)
        # 池化后 PAD 位置已经和短语特征混在一起，无法再精确掩码，直接对全序列取最大值
        over_time = states.max(dim=1).values
        return self.decoder(self.dropout(torch.cat([last, over_time], dim=1)))


def main() -> None:
    parser = common.build_parser(epochs=10, batch_size=64, lr=1e-3)
    parser.add_argument("--num-filter", type=int, default=128)
    parser.add_argument("--filter-size", type=int, default=3)
    parser.add_argument("--pooling-size", type=int, default=2)
    parser.add_argument("--num-hiddens", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    args = parser.parse_args()

    common.run(
        NAME, args,
        lambda bundle: SentimentNet(bundle.weight, num_filter=args.num_filter,
                                    filter_size=args.filter_size,
                                    pooling_size=args.pooling_size,
                                    num_hiddens=args.num_hiddens,
                                    num_layers=args.num_layers, dropout=args.dropout),
        lambda model: optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr),
    )


if __name__ == "__main__":
    main()
