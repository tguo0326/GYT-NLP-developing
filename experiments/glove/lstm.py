"""任务 8：双向 LSTM 情感分类（GloVe 词向量）。

三个要理解的概念：

* **隐藏状态 h_t**——「读到第 t 个词时的短期理解」。每一步由上一步的 h_{t-1} 和
  当前词共同决定，是 LSTM 对外输出的部分（`states`）。
* **记忆单元 c_t**——「贯穿整句的长期记忆」。c_t 只被遗忘门乘、被输入门加，
  梯度沿 c 这条路径几乎是恒等映射传播，这才是 LSTM 能处理长依赖、
  而普通 RNN 会梯度消失的原因。三个门（遗忘/输入/输出）都是对 c 的读写控制。
* **双向**——正向读一遍、反向再读一遍，两个方向的 h_t 拼起来。
  `it was not until the ending that the film became bearable` 这种句子，
  只正向读到 `not` 时还不知道否定的是什么；反向那一遍提供了右侧上下文。

相比原始版本（docs/original_code/imdb_lstm.py）的关键修改：

* **用 pack_padded_sequence 处理填充**。原代码取 `states[-1]`——序列被填到 512，
  而评论中位数只有 174 词，正向 LSTM 的「最后一步」其实读了三百多个 PAD，
  真实句尾的信息早被冲掉了。打包后 LSTM 只跑有效长度，h_n 就是真正的句尾状态；
* 句子表示 = 双向最终隐藏状态 ⊕ 时间轴最大池化（各 2×hidden，合计 4×hidden，
  与原代码的输出维度一致）。最大池化补上「哪个词最强」这一路信息；
* dropout：原代码写 `dropout=0` 且 num_layers=2，等于两层之间毫无正则；
* Adam(lr=0.05) → Adam(lr=1e-3)。0.05 对 LSTM 太大，第一个 epoch 就会发散到 0.5 准确率；
* 设备选择、种子、train()/eval()、no_grad、最佳模型保存全部移交 common.py。

用法：

    python experiments/glove/lstm.py
    python experiments/glove/lstm.py --num-hiddens 120 --num-layers 2
    python experiments/glove/lstm.py --predict "The pacing dragged and I nearly fell asleep."
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

NAME = "lstm"


class SentimentNet(nn.Module):
    def __init__(self, weight: torch.Tensor, num_hiddens: int = 120, num_layers: int = 2,
                 bidirectional: bool = True, labels: int = 2, dropout: float = 0.5):
        super().__init__()
        embed_size = weight.shape[1]
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        # 最终隐藏状态 (dir×hidden) ⊕ 时间轴最大池化 (dir×hidden)
        self.decoder = nn.Linear(num_hiddens * self.num_directions * 2, labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # PAD 的 id 是 0，据此还原每条评论的真实长度
        lengths = (inputs != 0).sum(dim=1).clamp(min=1)
        embeddings = self.embedding(inputs)

        packed = pack_padded_sequence(embeddings, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed_states, (hidden, _cell) = self.encoder(packed)
        states, _ = pad_packed_sequence(packed_states, batch_first=True)

        # hidden: (num_layers × num_directions, batch, hidden)，取最后一层的各个方向
        last = hidden[-self.num_directions:].permute(1, 0, 2).flatten(1)

        # 最大池化前把 PAD 位置压成 -inf，否则会被 0 抬高
        mask = (inputs[:, :states.shape[1]] != 0).unsqueeze(2)
        pooled = states.masked_fill(~mask, float("-inf")).max(dim=1).values

        return self.decoder(self.dropout(torch.cat([last, pooled], dim=1)))


def main() -> None:
    parser = common.build_parser(epochs=10, batch_size=64, lr=1e-3)
    parser.add_argument("--num-hiddens", type=int, default=120)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--unidirectional", action="store_true", help="关闭双向，做对照实验")
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
