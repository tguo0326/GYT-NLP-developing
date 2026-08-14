"""任务 9：双向 GRU 情感分类（GloVe 词向量）。

GRU 与 LSTM 的区别，一句话：**GRU 把 LSTM 的三个门压成两个，并且取消了独立的
记忆单元 c**。

| | LSTM | GRU |
|---|---|---|
| 门 | 遗忘门 f、输入门 i、输出门 o | 重置门 r、更新门 z |
| 状态 | 隐藏状态 h + 记忆单元 c | 只有 h |
| 单层单向参数量 | 4 × (input+hidden+1) × hidden | 3 × (input+hidden+1) × hidden |

LSTM 用「遗忘多少 + 写入多少」两个独立的门控制 c；GRU 用一个更新门 z 同时决定
「保留多少旧 h、写入多少新 h」，二者互补（`h_t = (1-z)·h_{t-1} + z·h̃_t`）。
少一个门 ⇒ 权重矩阵少 1/4 ⇒ 参数少约 25%、训练也快一档。代价是表达能力略弱，
但在 IMDB 这种句子长度几百词、任务本身不算难的场景，两者准确率通常差不到 1 个点。

**本脚本与 imdb_lstm.py 结构、超参数、随机种子完全一致，只把 nn.LSTM 换成 nn.GRU**，
这样 results/ 里的参数量、训练时间、验证准确率三项才是可比的。

相比原始版本（docs/original_code/imdb_gru.py）的修改与 imdb_lstm.py 相同：
pack_padded_sequence 处理填充、加 dropout、SGD(lr=0.8) → Adam(lr=1e-3)、
训练循环移交 common.py。

用法：

    python experiments/glove/gru.py
    python experiments/glove/gru.py --predict "One of the finest performances I have ever seen."
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许 `python experiments/.../x.py` 直接运行（不必写成 python -m）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from core import common

NAME = "gru"


class SentimentNet(nn.Module):
    def __init__(self, weight: torch.Tensor, num_hiddens: int = 120, num_layers: int = 2,
                 bidirectional: bool = True, labels: int = 2, dropout: float = 0.5):
        super().__init__()
        embed_size = weight.shape[1]
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=0)
        # 与 LSTM 版唯一的差别就是这一行
        self.encoder = nn.GRU(
            input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(num_hiddens * self.num_directions * 2, labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lengths = (inputs != 0).sum(dim=1).clamp(min=1)
        embeddings = self.embedding(inputs)

        packed = pack_padded_sequence(embeddings, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        # GRU 只返回 h，没有 LSTM 的 (h, c) 元组——这是两者最直观的接口差异
        packed_states, hidden = self.encoder(packed)
        states, _ = pad_packed_sequence(packed_states, batch_first=True)

        last = hidden[-self.num_directions:].permute(1, 0, 2).flatten(1)
        mask = (inputs[:, :states.shape[1]] != 0).unsqueeze(2)
        pooled = states.masked_fill(~mask, float("-inf")).max(dim=1).values

        return self.decoder(self.dropout(torch.cat([last, pooled], dim=1)))


def main() -> None:
    parser = common.build_parser(epochs=10, batch_size=64, lr=1e-3)
    parser.add_argument("--num-hiddens", type=int, default=120)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--unidirectional", action="store_true")
    args = parser.parse_args()

    common.run(
        NAME, args,
        lambda bundle: SentimentNet(bundle.weight, num_hiddens=args.num_hiddens,
                                    num_layers=args.num_layers,
                                    bidirectional=not args.unidirectional,
                                    dropout=args.dropout),
        lambda model: optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr),
    )


if __name__ == "__main__":
    main()
