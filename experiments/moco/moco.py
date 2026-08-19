"""SCL-MoCo 的两个核心部件：特征队列 与 动量编码器。

参照 He et al., Momentum Contrast for Unsupervised Visual Representation Learning
(CVPR 2020, https://arxiv.org/abs/1911.05722)，迁移其中两件事：

1. **历史特征队列**：把过去若干步的 key 特征存下来当候选，
   让每一步能对比的样本数从「真实 batch」放大到 `queue_size`；
2. **动量编码器**：队列里的特征来自不同时刻的参数，如果那些参数是被梯度
   剧烈更新的，队列里的特征就彼此不一致。MoCo 的解法是让 key 分支的参数
   用 EMA 缓慢跟随 query 分支：`θ_k ← m·θ_k + (1-m)·θ_q`。

**显存关键点**：这里不复制第二套 DeBERTa-v3-large。底座本来就是冻结的，
所以动量副本只需要覆盖真正会变的那些参数 —— LoRA 矩阵、pooler、projection head、
分类头。实现方式是在**同一个冻结底座**上挂第二套 peft adapter（`k_enc`），
peft 的 `modules_to_save` 机制会自动为每个 adapter 各存一份 pooler/proj_head/classifier。
代价约 4 M 参数（≈16 MB fp32），而复制整个底座是 +435 M 参数（≈1.7 GB）。
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

QUERY_ADAPTER = "q_enc"
KEY_ADAPTER = "k_enc"


# ---------------------------------------------------------------- projection head
class ProjectionHead(nn.Module):
    """MoCo v2 / SimCLR 式的两层 MLP 投影头：hidden → hidden → proj_dim。

    对比损失作用在投影后的低维空间上，分类头仍然吃池化后的句向量 ——
    这样对比任务不会直接改写分类用的表示，是 SimCLR 那篇的主要发现之一。
    """

    def __init__(self, hidden_size: int, proj_dim: int = 128):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.out = nn.Linear(hidden_size, proj_dim)

    def forward(self, features):
        return self.out(self.activation(self.dense(features)))


def masked_mean_pool(hidden_states, attention_mask):
    """mask 平均池化。DeBERTa 的 `outputs[1]` 不是句向量（没有 BERT 那种 pooler
    输出在 tuple 里），而且 padding 位置必须排除，所以统一自己池化。"""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


# ------------------------------------------------------------------ feature queue
class FeatureQueue(nn.Module):
    """固定长度 FIFO 队列，存 L2 归一化后的特征 + 真实标签。

    全部字段都是 buffer，所以 `state_dict()` 天然带上它们，
    checkpoint 保存/恢复不需要额外代码。

    - `feats`  [K, d]  归一化特征
    - `labels` [K]     对应的真实标签，未填充的槽位是 -1
    - `ptr`    []      下一个写入位置
    - `valid`  []      已填充的有效长度（上限 K）

    初始队列**不填随机特征和随机标签**：`feats` 全 0、`labels` 全 -1，
    并且靠 `valid` 屏蔽，`usable_mask()` 只放行前 `valid` 个槽位。
    """

    def __init__(self, queue_size: int, dim: int):
        super().__init__()
        self.queue_size = int(queue_size)
        self.dim = int(dim)
        self.register_buffer("feats", torch.zeros(self.queue_size, self.dim))
        self.register_buffer("labels", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("valid", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        """新特征入队、最旧特征被覆盖。batch 跨过队尾时分两段写。"""
        feats = feats.detach().to(self.feats.dtype).to(self.feats.device)
        labels = labels.detach().to(self.labels.device).long()
        n = feats.shape[0]
        if n == 0:
            return
        if n >= self.queue_size:            # 一个 batch 就填满整个队列（极端情况）
            self.feats.copy_(feats[-self.queue_size:])
            self.labels.copy_(labels[-self.queue_size:])
            self.ptr.fill_(0)
            self.valid.fill_(self.queue_size)
            return
        start = int(self.ptr.item())
        end = start + n
        if end <= self.queue_size:
            self.feats[start:end] = feats
            self.labels[start:end] = labels
        else:                               # 绕回队头
            head = self.queue_size - start
            self.feats[start:] = feats[:head]
            self.labels[start:] = labels[:head]
            self.feats[:end - self.queue_size] = feats[head:]
            self.labels[:end - self.queue_size] = labels[head:]
        self.ptr.fill_(end % self.queue_size)
        self.valid.fill_(min(self.queue_size, int(self.valid.item()) + n))

    def usable_mask(self) -> torch.Tensor:
        """[K] 的 0/1 掩码：只有前 valid 个槽位可用。"""
        mask = torch.zeros(self.queue_size, device=self.feats.device)
        mask[: int(self.valid.item())] = 1.0
        return mask

    def label_counts(self) -> dict:
        valid = int(self.valid.item())
        if valid == 0:
            return {}
        labels = self.labels[:valid]
        return {int(k): int((labels == k).sum().item()) for k in labels.unique()}

    def extra_repr(self) -> str:
        return f"queue_size={self.queue_size}, dim={self.dim}"


# --------------------------------------------------------------- momentum encoder
class MomentumBranch(nn.Module):
    """动量 key 编码器：共享冻结底座，只维护第二套 adapter 参数。

    对外只有三个动作：
        `hard_sync()`      —— 把 key 参数硬拷贝成 query 参数（初始化时用）
        `ema_update(m)`    —— θ_k ← m·θ_k + (1-m)·θ_q（**每个 optimizer step 一次**）
        `encode(inputs)`   —— 用 key 分支前向，返回归一化特征（全程 no_grad）

    key 参数一律 `requires_grad=False`，且不会被放进 optimizer
    （见 `run_experiment.py` 里对 `optimizer.param_groups` 的断言）。
    """

    def __init__(self, model, feature_fn, query_adapter=QUERY_ADAPTER,
                 key_adapter=KEY_ADAPTER):
        super().__init__()
        self.model = [model]              # 放进 list 避免 nn.Module 重复注册子模块
        self.feature_fn = feature_fn
        self.query_adapter = query_adapter
        self.key_adapter = key_adapter
        self.pairs = self._match_params(model)
        if not self.pairs:
            raise RuntimeError("没有匹配到任何 query/key 参数对，检查 adapter 名字")
        n_param = sum(q.numel() for q, _ in self.pairs)
        logger.info("动量分支：%d 组参数对，共 %s 个参数（%.1f MB fp32）",
                    len(self.pairs), f"{n_param:,}", n_param * 4 / 1024 ** 2)

    def _match_params(self, model):
        named = dict(model.named_parameters())
        pairs = []
        for name, param in named.items():
            if f".{self.query_adapter}." not in name and \
                    not name.endswith(f".{self.query_adapter}"):
                continue
            key_name = name.replace(f".{self.query_adapter}.", f".{self.key_adapter}.")
            if key_name == name:
                key_name = name[: -len(self.query_adapter)] + self.key_adapter
            if key_name not in named:
                raise RuntimeError(f"{name} 找不到对应的 key 参数 {key_name}")
            if named[key_name].shape != param.shape:
                raise RuntimeError(f"{name} 与 {key_name} 形状不一致")
            pairs.append((param, named[key_name]))
        return pairs

    # ---- 参数同步 ----
    @torch.no_grad()
    def hard_sync(self) -> None:
        for q, k in self.pairs:
            k.data.copy_(q.data)

    @torch.no_grad()
    def ema_update(self, m: float) -> None:
        for q, k in self.pairs:
            k.data.mul_(m).add_(q.data.detach(), alpha=1.0 - m)

    def max_abs_diff(self) -> float:
        """query 与 key 参数的最大绝对差，smoke test 用。"""
        with torch.no_grad():
            return max((q.data - k.data).abs().max().item() for q, k in self.pairs)

    def freeze_key(self) -> None:
        for _, k in self.pairs:
            k.requires_grad_(False)

    def key_params(self):
        return [k for _, k in self.pairs]

    def query_params(self):
        return [q for q, _ in self.pairs]

    # ---- 前向 ----
    @torch.no_grad()
    def encode(self, inputs: dict) -> torch.Tensor:
        """切到 key adapter 跑一次前向，返回 [B, proj_dim] 的归一化特征。

        全程 `torch.no_grad()`：key 分支不参与反向传播，也不保留 activation，
        所以显存开销只有一次前向的瞬时占用。
        """
        model = self.model[0]
        model.set_adapter(self.key_adapter)
        self.freeze_key()
        try:
            feats = self.feature_fn(inputs)
        finally:
            # peft 的 set_adapter 会顺便把 active adapter 的 requires_grad 打开、
            # 其余关掉，所以切回来之后 query 可训练、key 冻结，正是我们要的状态。
            model.set_adapter(self.query_adapter)
            self.freeze_key()
        return F.normalize(feats.float(), dim=-1).detach()


class MomentumCallbackState:
    """记录 EMA 实际被调用了多少次，用于验证「跟随 optimizer step 而不是 micro-batch」。"""

    def __init__(self):
        self.ema_calls = 0
        self.enqueue_calls = 0
        self.enqueued_items = 0
