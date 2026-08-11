"""任务 11（选做）：Capsule Network + 双向 LSTM（GloVe 词向量）。

胶囊网络（Sabour et al. 2017）的核心是**用向量而不是标量表示特征**：
一个胶囊输出的向量，长度代表「这个特征存在的置信度」，方向代表「特征的具体形态」。
配上动态路由（dynamic routing），低层胶囊会把自己的输出更多地送给
「与自己预测一致」的高层胶囊——相当于一种可迭代的软聚类。

放在文本上：LSTM 每个时间步的隐藏状态是一个低层胶囊，动态路由把 512 个时间步
聚合成 `num_capsule` 个高层胶囊。和最大池化（硬性取一个位置）、
注意力（一组权重加权求和）相比，路由做的是**多次迭代的、多中心的**聚合。

相比原始版本（docs/original_code/imdb_capsule_lstm.py）修的问题：

1. **形状不一致**：`W` 的输出维度是 `num_capsule * num_hiddens*2`（5×256=1280），
   紧接着 `view(batch, seq, num_capsule=5, dim_capsule=5)` 要求 25。跑起来直接
   RuntimeError。这里让 `W` 输出 `num_capsule * dim_capsule`；
2. **squash 写错了**：原代码 `x / sqrt(‖s‖²+ε)` 只是 L2 归一化，所有胶囊长度都变成 1，
   「置信度」这一路信息被抹平。正确的 squash 是
   `‖s‖²/(1+‖s‖²) · s/‖s‖`，长度被压到 (0,1) 且保留大小关系；
3. **路由权重 b 的更新在 no_grad 之外**：原代码只把初始化包进 `no_grad`，
   之后 `b = sum(outputs * u_hat)` 会把整个路由迭代都记进计算图。
   路由是「前向的软聚类」，按原论文不应该反传，这里用 `.detach()`；
4. **`capsule[0]` / `capsule[-1]` 索引错了维度**：capsule 的形状是
   `(batch, num_capsule, dim_capsule)`，取 `[0]` / `[-1]` 拿的是 batch 里的
   第一个和最后一个样本。正确做法是把所有胶囊展平后送进分类器；
5. LSTM 没有 `batch_first`，states 是 `(seq, batch, hidden)`，
   却当成 `(batch, seq, hidden)` 用；
6. 删掉 forward 里的三个 `print`（每个 batch 都打，日志会爆）。

用法：

    python imdb_capsule_lstm.py
    python imdb_capsule_lstm.py --num-capsule 8 --dim-capsule 16 --routings 3
"""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.nn import functional as F

import common

NAME = "capsule_lstm"


class Capsule(nn.Module):
    """带动态路由的胶囊层。输入 (batch, seq, in_dim) → 输出 (batch, num_capsule, dim_capsule)。"""

    def __init__(self, in_dim: int, num_capsule: int = 8, dim_capsule: int = 16,
                 routings: int = 3):
        super().__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        # 每个输入位置经过 W 得到 num_capsule 个「预测向量」u_hat
        self.W = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(1, in_dim, num_capsule * dim_capsule)))

    @staticmethod
    def squash(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-7) -> torch.Tensor:
        """把向量长度压到 (0, 1) 而保留方向和相对大小——胶囊的「概率」就是长度。"""
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + eps)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = inputs.shape
        u_hat = torch.matmul(inputs, self.W)                     # (batch, seq, nc*dc)
        u_hat = u_hat.view(batch, seq, self.num_capsule, self.dim_capsule)
        u_hat = u_hat.permute(0, 2, 1, 3)                        # (batch, nc, seq, dc)

        logits = torch.zeros(batch, self.num_capsule, seq, device=inputs.device)
        if mask is not None:
            # PAD 位置不参与路由，否则几百个填充步会稀释真实词的贡献
            logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))

        outputs = None
        for iteration in range(self.routings):
            # 沿输入位置归一化：每个位置决定把自己的预测分给哪些高层胶囊
            coupling = F.softmax(logits, dim=2)
            outputs = self.squash((coupling.unsqueeze(-1) * u_hat).sum(dim=2))
            if iteration < self.routings - 1:
                # 一致性越高（内积越大）的连接，下一轮权重越大。
                # detach：路由是前向的软聚类过程，不参与反向传播。
                agreement = (outputs.unsqueeze(2) * u_hat).sum(dim=-1)
                logits = logits + agreement.detach()
        return outputs


class SentimentNet(nn.Module):
    def __init__(self, weight: torch.Tensor, num_hiddens: int = 128, num_layers: int = 2,
                 bidirectional: bool = True, num_capsule: int = 8, dim_capsule: int = 16,
                 routings: int = 3, labels: int = 2, dropout: float = 0.5):
        super().__init__()
        embed_size = weight.shape[1]
        num_directions = 2 if bidirectional else 1
        hidden_dim = num_hiddens * num_directions

        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.capsule = Capsule(hidden_dim, num_capsule, dim_capsule, routings)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(num_capsule * dim_capsule, labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mask = inputs != 0
        states, _ = self.encoder(self.embedding(inputs))
        capsules = self.capsule(states, mask)
        # 展平所有胶囊：形状 (batch, num_capsule × dim_capsule)
        return self.decoder(self.dropout(capsules.flatten(1)))


def main() -> None:
    parser = common.build_parser(epochs=10, batch_size=64, lr=1e-3)
    parser.add_argument("--num-hiddens", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-capsule", type=int, default=8)
    parser.add_argument("--dim-capsule", type=int, default=16)
    parser.add_argument("--routings", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    args = parser.parse_args()

    common.run(
        NAME, args,
        lambda bundle: SentimentNet(bundle.weight, num_hiddens=args.num_hiddens,
                                    num_layers=args.num_layers,
                                    num_capsule=args.num_capsule,
                                    dim_capsule=args.dim_capsule,
                                    routings=args.routings, dropout=args.dropout),
        lambda model: optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr),
    )


if __name__ == "__main__":
    main()
