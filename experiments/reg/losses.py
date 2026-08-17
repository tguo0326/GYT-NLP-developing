"""
正则化损失函数集合。

- SupConLoss: 监督对比学习损失，原始实现来自
  Author: Yonglong Tian (yonglong@mit.edu), Date: May 07, 2020
  论文: Supervised Contrastive Learning, https://arxiv.org/abs/2004.11362
- SCLLoss: 在 SupConLoss 外面包一层，负责 L2 归一化和 view 维度补齐，
  直接吃 [bsz, dim] 的句向量。NLP 侧的用法见
  Gunel et al., Supervised Contrastive Learning for Pre-trained Language
  Model Fine-tuning, https://arxiv.org/abs/2011.01403
- rdrop_kl_loss: R-Drop 的对称 KL 项
  论文: R-Drop: Regularized Dropout for Neural Networks,
  https://arxiv.org/abs/2106.14448
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if features.dim() < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], '
                             'at least 3 dimensions are required')
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
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        # 一个 batch 里某一类只出现一次时 mask.sum(1) == 0，会算出 nan，
        # 这里把这些 anchor 直接丢掉（原始实现没处理，长尾 batch 上会炸）
        pos_per_anchor = mask.sum(1)
        valid = pos_per_anchor > 0
        if not valid.any():
            return anchor_dot_contrast.sum() * 0.0

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid] / pos_per_anchor[valid]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


class SCLLoss(nn.Module):
    """SupConLoss 的易用封装：吃 [bsz, dim] 的句向量。

    对比学习要求特征落在单位球面上（余弦相似度），所以必须先做 L2 归一化，
    否则除以 temperature=0.07 之后 logits 量级会爆掉。
    """

    def __init__(self, temperature=0.3, base_temperature=0.3):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature,
                                 base_temperature=base_temperature)

    def forward(self, features, labels):
        features = F.normalize(features.float(), dim=-1)
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [bsz, dim] -> [bsz, 1, dim]
        return self.supcon(features, labels)


def rdrop_kl_loss(logits_p, logits_q, reduction="batchmean"):
    """R-Drop 的对称 KL 散度项: 0.5 * (KL(p||q) + KL(q||p))。

    两路 logits 来自同一批样本的两次前向，因为 dropout 采样不同而不同。
    用 log_target=True 走 log 空间，数值上比先 softmax 再取 log 更稳。
    reduction 用 batchmean 而不是 sum，保证 loss 量级与 batch size 无关。
    """
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    kl_pq = F.kl_div(log_p, log_q, reduction=reduction, log_target=True)
    kl_qp = F.kl_div(log_q, log_p, reduction=reduction, log_target=True)
    return 0.5 * (kl_pq + kl_qp)
