"""任务 10-B：Attention + 双向 LSTM，并把注意力权重打印出来。

**早期 Attention 解决的是什么问题？** 纯 LSTM 必须把整条评论压进一个固定长度的
向量（最终隐藏状态），几百个词的信息全挤在 240 维里，靠后的词天然占优势——
这就是「信息瓶颈」。而最大池化虽然缓解了这点，却是硬性的 argmax，只留下一个位置。

Attention 的做法是让模型自己学一个打分函数：给每个时间步的隐藏状态 h_t 算一个
标量分数，softmax 归一化成权重 α_t，句子表示 = Σ α_t · h_t。
这里用的是 Yang et al. 2016（Hierarchical Attention Networks）的加性打分：

    u_t = tanh(W · h_t)          # 先做一次非线性变换
    e_t = u_tᵀ · v              # 和一个可学习的「查询向量」v 做内积
    α   = softmax(e)             # 沿时间轴归一化
    c   = Σ α_t · h_t            # 加权求和

这被称为「self-attention 的前身」：查询向量 v 是全局共享的静态参数，而不是像
Transformer 那样由每个位置自己生成 Q/K/V。所以它只能表达「哪些词对这个任务重要」，
不能表达「哪些词对彼此重要」。副产品是可解释性——α 直接告诉你模型在看哪些词。

相比原始版本（docs/original_code/imdb_attention_lstm.py）修了两个实质性 bug：

1. `F.softmax(att, dim=1)`：原代码的 states 形状是 `(seq_len, batch, hidden)`
   （没用 batch_first），dim=1 是 **batch 维**。归一化跑到了「同一个 batch 里的
   不同样本」之间——batch_size 一改结果就变，而且完全没有意义；
2. `outputs = x * att_score` 之后没有求和，Attention 层返回的还是整条序列，
   接着又走 `torch.cat([states[0], states[-1]])` 取首尾——加权算完就被丢掉了。

另外补上 PAD 掩码（不掩码时 softmax 会把概率分给几百个填充位）、
把注意力权重导出到 results/，其余（设备、种子、日志、最佳模型）交给 common.py。

用法：

    python imdb_attention_lstm.py
    python imdb_attention_lstm.py --show-attention 6 --top-words 12
    python imdb_attention_lstm.py --predict "The ending ruined an otherwise great film."
"""

from __future__ import annotations

import json
import logging

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import common

NAME = "attention_lstm"


class Attention(nn.Module):
    """加性（Bahdanau 风格）注意力打分，沿时间轴归一化。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_omega = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.u_omega = nn.Parameter(torch.empty(hidden_dim, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """states: (batch, seq, hidden)，mask: (batch, seq) 布尔，True 表示真实词。

        返回 (上下文向量 (batch, hidden), 注意力权重 (batch, seq))。
        """
        scores = torch.matmul(torch.tanh(torch.matmul(states, self.w_omega)), self.u_omega)
        scores = scores.squeeze(2).masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)          # dim=1 是时间轴（batch_first=True）
        context = torch.bmm(weights.unsqueeze(1), states).squeeze(1)
        return context, weights


class SentimentNet(nn.Module):
    def __init__(self, weight: torch.Tensor, num_hiddens: int = 128, num_layers: int = 2,
                 bidirectional: bool = True, labels: int = 2, dropout: float = 0.5):
        super().__init__()
        embed_size = weight.shape[1]
        self.num_directions = 2 if bidirectional else 1
        hidden_dim = num_hiddens * self.num_directions

        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # 注意力上下文 ⊕ 双向最终隐藏状态，维度与原代码的 num_hiddens*4 一致
        self.decoder = nn.Linear(hidden_dim * 2, labels)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (logits, 注意力权重)。权重形状 (batch, seq_valid)。"""
        lengths = (inputs != 0).sum(dim=1).clamp(min=1)
        embeddings = self.embedding(inputs)

        packed = pack_padded_sequence(embeddings, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed_states, (hidden, _cell) = self.encoder(packed)
        states, _ = pad_packed_sequence(packed_states, batch_first=True)

        mask = inputs[:, :states.shape[1]] != 0
        context, weights = self.attention(states, mask)
        last = hidden[-self.num_directions:].permute(1, 0, 2).flatten(1)

        logits = self.decoder(self.dropout(torch.cat([context, last], dim=1)))
        return logits, weights

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encode(inputs)[0]


@torch.no_grad()
def explain(model: SentimentNet, features: torch.Tensor, bundle: common.Bundle,
            device: torch.device, top_words: int = 10) -> list[dict]:
    """对给定样本导出注意力权重最高的词——任务 10 的「展示模型重点关注的词语」。"""
    model.eval()
    logits, weights = model.encode(features.to(device))
    probs = torch.softmax(logits, dim=1).cpu()
    weights = weights.cpu()

    reports = []
    for row in range(features.shape[0]):
        ids = features[row]
        length = int((ids != 0).sum())
        tokens = [bundle.idx_to_word.get(int(i), "<unk>") for i in ids[:length]]
        scores = weights[row, :length]
        order = torch.argsort(scores, descending=True)[:top_words]
        reports.append({
            "prob_positive": round(float(probs[row, 1]), 4),
            "prediction": "positive" if int(probs[row].argmax()) == 1 else "negative",
            "length": length,
            "top_words": [(tokens[i], round(float(scores[i]), 5)) for i in order.tolist()],
            "excerpt": " ".join(tokens[:40]),
        })
    return reports


def main() -> None:
    parser = common.build_parser(epochs=10, batch_size=64, lr=1e-3)
    parser.add_argument("--num-hiddens", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--show-attention", type=int, default=4,
                        help="训练后导出多少条验证集样本的注意力权重")
    parser.add_argument("--top-words", type=int, default=10)
    args = parser.parse_args()

    result = common.run(
        NAME, args,
        lambda bundle: SentimentNet(bundle.weight, num_hiddens=args.num_hiddens,
                                    num_layers=args.num_layers, dropout=args.dropout),
        lambda model: optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr),
    )
    model, bundle, device = result.model, result.bundle, result.device

    logging.info("=== 注意力权重（验证集前 %d 条） ===", args.show_attention)
    samples = bundle.val_features[: args.show_attention]
    truths = bundle.val_labels[: args.show_attention].tolist()
    reports = explain(model, samples, bundle, device, args.top_words)
    for report, truth in zip(reports, truths):
        report["truth"] = "positive" if truth == 1 else "negative"
        logging.info("预测 %s (p=%.4f) / 真实 %s", report["prediction"],
                     report["prob_positive"], report["truth"])
        logging.info("  开头: %s ...", report["excerpt"])
        logging.info("  高权重词: %s",
                     ", ".join(f"{word}({score:.4f})" for word, score in report["top_words"]))

    # 对自定义评论也导出一份，方便直观检查模型是不是真的在看情感词
    demo = common.encode_texts(common.DEMO_REVIEWS, bundle.word_to_idx)
    demo_reports = explain(model, demo, bundle, device, args.top_words)
    for text, report in zip(common.DEMO_REVIEWS, demo_reports):
        report["review"] = text
        logging.info("[%s p=%.4f] %s", report["prediction"], report["prob_positive"], text)
        logging.info("  高权重词: %s",
                     ", ".join(f"{word}({score:.4f})" for word, score in report["top_words"]))

    path = common.RESULTS_DIR / f"{NAME}_attention.json"
    path.write_text(json.dumps({"validation": reports, "custom": demo_reports},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("注意力权重已写出 %s", path.name)


if __name__ == "__main__":
    main()
