"""监督对比损失：普通 SCL 与 SCL-MoCo 共用同一套数学。

三个东西：

- `SupConLoss`：Yonglong Tian 的官方实现（https://arxiv.org/abs/2004.11362），
  从 `new work 3/losses.py` 原样搬过来，**只作为正确性基准**用于 smoke test
  比对，训练路径不走它。
- `supcon_core`：本文件的核心。query 集合 × 候选集合 的监督对比损失，
  候选可以是「当前 batch 自己」（→ 普通 SCL）或「动量 key + 历史队列」（→ SCL-MoCo）。
  两条路径共用同一个函数，保证 SCL 与 SCL-MoCo 的差别**只有候选集合来自哪里**。
- `ContrastiveStats`：每步的正/负样本数量等统计，用于日志举证。

数学（SupCon 论文的 L_out 形式，把求和放在 log 外面）：

    L_i = -1/|P(i)| * Σ_{p∈P(i)} log [ exp(q_i·c_p/τ) / Σ_{a∈A(i)} exp(q_i·c_a/τ) ]

    P(i) = 候选里与 i 同标签的；A(i) = 全部可用候选（正负都在分母里）。
    无正样本的 query 直接从 loss 里剔除（不是填 0，那样会把梯度稀释）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """官方 SupConLoss（仅作 smoke test 的对照基准，训练不用）。

    相对原版的两处修补沿用 `new work 3/losses.py`：分母 `+1e-12`；
    `mask.sum(1)==0` 的 anchor 丢掉而不是除零得 nan。
    """

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        if features.dim() < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature, anchor_count = features[:, 0], 1
        elif self.contrast_mode == 'all':
            anchor_feature, anchor_count = contrast_feature, contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_per_anchor = mask.sum(1)
        valid = pos_per_anchor > 0
        if not valid.any():
            return anchor_dot_contrast.sum() * 0.0
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid] / pos_per_anchor[valid]
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


@dataclass
class ContrastiveStats:
    """一步对比损失的统计量，全部是 python 标量，方便直接写日志。"""

    n_query: int = 0                 # 本步 query 总数（= 真实 micro-batch）
    n_query_with_pos: int = 0        # 至少有一个正样本、真正参与 loss 的 query 数
    n_candidates: int = 0            # 每个 query 可见的候选总数（batch + 队列）
    pos_per_query: float = 0.0       # 平均正样本数（只在有正样本的 query 上平均）
    neg_per_query: float = 0.0       # 平均负样本数（同上）
    queue_valid: int = 0             # 有效队列长度
    queue_label_counts: dict = field(default_factory=dict)  # 队列标签分布
    # 逐 query 的正/负样本数，供 smoke test 做「两组严格一致」的比对
    pos_counts: tuple = ()
    neg_counts: tuple = ()

    def as_log_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}n_query": self.n_query,
            f"{prefix}n_query_with_pos": self.n_query_with_pos,
            f"{prefix}n_candidates": self.n_candidates,
            f"{prefix}pos_per_query": round(self.pos_per_query, 3),
            f"{prefix}neg_per_query": round(self.neg_per_query, 3),
            f"{prefix}queue_valid": self.queue_valid,
        }


def supcon_core(query, cand, query_labels, cand_labels, temperature,
                base_temperature=None, cand_usable=None, self_index=None):
    """监督对比损失，返回 (loss, ContrastiveStats)。

    Args:
        query:        [B, d] 已 L2 归一化
        cand:         [N, d] 已 L2 归一化（普通 SCL 时就是 query 本身）
        query_labels: [B]
        cand_labels:  [N]，无效槽位可以是任意值（靠 cand_usable 屏蔽）
        temperature:  τ
        base_temperature: 默认等于 τ（则前置系数为 1，与 new work 3 的 SCL 一致）
        cand_usable:  [N] bool/0-1，队列里未填满的槽位为 0
        self_index:   [B] long，query i 在 cand 里对应「自己那一条」的下标，
                      传了就把该位置从候选中剔除（普通 SCL 去对角线用）。
                      SCL-MoCo 默认**不**传：MoCo 里 k_i 是同一样本的另一个视角
                      （dropout + EMA 滞后造成差异），按原论文当正样本用。
    """
    base_temperature = temperature if base_temperature is None else base_temperature
    query = query.float()
    cand = cand.float()
    n_query, n_cand = query.shape[0], cand.shape[0]
    device = query.device

    usable = torch.ones(n_query, n_cand, device=device)
    if cand_usable is not None:
        usable = usable * cand_usable.to(device=device, dtype=usable.dtype).view(1, -1)
    if self_index is not None:
        usable = usable.scatter(1, self_index.view(-1, 1).to(device), 0.0)

    logits = (query @ cand.t()) / temperature
    # 数值稳定：减掉每行（可用候选中的）最大值。softmax 对整体平移不变，
    # 不改变数学结果；detach 是因为这只是常数偏移。
    neg_inf = torch.finfo(logits.dtype).min
    row_max = logits.masked_fill(usable == 0, neg_inf).max(dim=1, keepdim=True).values
    row_max = torch.nan_to_num(row_max, neginf=0.0).detach()
    logits = logits - row_max

    pos_mask = (query_labels.view(-1, 1) == cand_labels.view(1, -1)).to(logits.dtype) * usable
    neg_mask = usable - pos_mask

    exp_logits = torch.exp(logits) * usable
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(1)
    valid = pos_count > 0

    neg_count = neg_mask.sum(1)
    stats = ContrastiveStats(
        n_query=n_query,
        n_query_with_pos=int(valid.sum().item()),
        n_candidates=int(usable[0].sum().item()) if n_query else 0,
        pos_per_query=float(pos_count[valid].mean().item()) if valid.any() else 0.0,
        neg_per_query=float(neg_count[valid].mean().item()) if valid.any() else 0.0,
        pos_counts=tuple(int(v) for v in pos_count.tolist()),
        neg_counts=tuple(int(v) for v in neg_count.tolist()),
    )

    if not valid.any():
        # 一个正样本都没有：返回与图连通的 0，绝不能 return 0.0（会断梯度）
        # 也不能除 0（会 nan）。
        return (query.sum() * 0.0), stats

    mean_log_prob_pos = (pos_mask * log_prob).sum(1)[valid] / pos_count[valid]
    loss = -(temperature / base_temperature) * mean_log_prob_pos.mean()
    return loss, stats


def scl_in_batch(features, labels, temperature, base_temperature=None):
    """单视图 SCL：候选就是当前 micro-batch 自己，去掉对角线。

    这是 new work 3 的老口径，**只保留给 smoke test 做数学等价性对照**
    （见 A1）。训练路径已经换成下面的双视图版本 —— 单视图时 batch 里某一类
    只出现一次的 query 完全没有正样本、会被整条丢掉（bs=4 实测约一半 micro-batch
    都会发生），那样它与 SCL-MoCo 之间就多出「有没有自身配对正样本」这个额外差异。
    """
    feats = F.normalize(features.float(), dim=-1)
    self_index = torch.arange(feats.shape[0], device=feats.device)
    return supcon_core(feats, feats, labels, labels, temperature,
                       base_temperature=base_temperature, self_index=self_index)


def scl_two_view(feats_v1, feats_v2, labels, temperature, base_temperature=None):
    """双视图 SCL：anchor = 视图1，候选 = 视图2 的全部 B 条。

    视图 2 由**同一个在线编码器**再前向一次得到（dropout 独立重采样），
    不用动量编码器、不用队列。于是：

        正样本 = 自身的第二视图 + batch 内其他同标签样本（的第二视图）
        负样本 = batch 内不同标签样本（的第二视图）

    候选集合的结构与 SCL-MoCo 的「本 batch 动量 key + 队列」在队列为空时**完全一致**
    （都是 B 条「另一视图」特征，自身那条算正样本），所以两组的差别被收窄到唯一一项：
    这 B 条来自在线编码器（带梯度）还是动量编码器（不带梯度）+ 是否再加上历史队列。
    """
    v1 = F.normalize(feats_v1.float(), dim=-1)
    v2 = F.normalize(feats_v2.float(), dim=-1)
    return supcon_core(v1, v2, labels, labels, temperature,
                       base_temperature=base_temperature)
