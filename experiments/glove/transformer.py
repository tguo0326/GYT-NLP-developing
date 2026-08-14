"""任务 11（选做）：Transformer Encoder 情感分类（GloVe 词向量）。

和 Attention-LSTM 的对比是这个脚本的意义所在：

* Attention-LSTM 里的注意力有**一个全局共享的查询向量** v，只能回答
  「哪些词对情感分类重要」；顺序信息完全由 LSTM 的递归结构提供。
* Transformer 的 self-attention 让**每个位置自己生成 Q/K/V**，回答的是
  「对第 i 个词来说，哪些词重要」——一个 n×n 的关系矩阵而不是一个 n 维权重向量。
  代价是完全丢掉了顺序，必须靠位置编码把「第几个词」重新注入。

相比原始版本（docs/original_code/imdb_transformer.py）改动很大，因为原代码有几处
让它根本跑不起来或跑出来没有意义的问题：

1. `review_to_wordlist` 返回的是 `' '.join(words)` 字符串，而 `Vocab.build` 对它做
   `for token in sentence`——**逐字符**建词表，词表退化成 26 个字母；
2. `nn.TransformerEncoderLayer(hidden_dim=120, ...)` 的 d_model 是 120，
   但送进去的是 300 维词向量，形状对不上；
3. `F.log_softmax` 的输出又喂给 `nn.CrossEntropyLoss`（内部再做一次 log_softmax），
   等于取了两次对数，梯度被严重压缩；
4. `hidden_states[0, :, :]` 取第一个位置当句子表示——但这里没有 `[CLS]` token，
   第一个位置只是评论的第一个词；
5. 验证循环 `net(val_feature)` 少传 `lengths` 参数，一进验证就 TypeError；
6. `train_test_split` 之后把 `train_labels` 覆盖成了划分后的标签，而 `train_reviews`
   里已经带了标签，标签实际用了两套。

这里改成复用 `pickle/imdb_glove.pickle3`（和 CNN/LSTM 完全同一份数据与划分），
d_model 直接取 300 对齐 GloVe 维度，句子表示用带掩码的平均池化。

用法：

    python experiments/glove/transformer.py
    python experiments/glove/transformer.py --num-layers 2 --num-heads 6 --batch-size 32
    python experiments/glove/transformer.py --predict "A slow burn that pays off in the final act."
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许 `python experiments/.../x.py` 直接运行（不必写成 python -m）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math

import torch
from torch import nn, optim

from core import common

NAME = "transformer"


class PositionalEncoding(nn.Module):
    """正弦位置编码（Vaswani et al. 2017）。

    self-attention 对输入顺序是置换等变的——打乱词序，输出只是跟着打乱，
    句子表示完全不变。位置编码就是把「第几个词」这一信息加回词向量里。
    用固定的正弦函数而非可学习参数，好处是能外推到训练时没见过的长度。
    """

    def __init__(self, d_model: int, max_len: int = common.MAX_LEN, dropout: float = 0.1):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        # register_buffer：随模型 to(device)/save 一起走，但不是可训练参数
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.encoding[:, : x.shape[1]])


class SentimentNet(nn.Module):
    def __init__(self, weight: torch.Tensor, num_heads: int = 6, num_layers: int = 2,
                 dim_feedforward: int = 512, labels: int = 2, dropout: float = 0.1):
        super().__init__()
        d_model = weight.shape[1]
        if d_model % num_heads:
            raise ValueError(f"d_model={d_model} 必须能被 num_heads={num_heads} 整除")

        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=0)
        self.scale = math.sqrt(d_model)
        self.positional = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pad_mask = inputs == 0                       # True 的位置要被注意力忽略
        # 乘 √d_model 是原论文的做法：让词向量的量级和位置编码可比
        hidden = self.positional(self.embedding(inputs) * self.scale)
        hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)

        # 带掩码的平均池化。没有 [CLS] token，就不能像 BERT 那样取第一个位置。
        valid = (~pad_mask).unsqueeze(2).float()
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.decoder(self.dropout(self.norm(pooled)))


def main() -> None:
    # Transformer 对学习率敏感，比 CNN/LSTM 低一档；序列 512 时显存也更吃紧
    parser = common.build_parser(epochs=10, batch_size=32, lr=3e-4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    common.run(
        NAME, args,
        lambda bundle: SentimentNet(bundle.weight, num_heads=args.num_heads,
                                    num_layers=args.num_layers,
                                    dim_feedforward=args.dim_feedforward,
                                    dropout=args.dropout),
        lambda model: optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01),
    )


if __name__ == "__main__":
    main()
