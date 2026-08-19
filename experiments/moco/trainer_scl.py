"""SCLMoCoTrainer：一个 Trainer 同时支持三种方法，保证除方法本身外没有任何差异。

    method=baseline   L = CE
    method=scl        L = CE + λ · SCL(anchor=视图1，候选=同一在线编码器的视图2，B 条)
    method=scl_moco   L = CE + λ · SCL(anchor=query，候选=动量 key B 条 + 历史队列 K 条)

两组的候选集合结构刻意做成一致的：都是「自身的另一个视图（算正样本）+ batch 内
其他样本的另一个视图」。于是**队列为空时两组每个 query 的候选数、以及按标签划分的
正负数量完全相同**，两组都不会因为缺正样本而 skip；差异被收窄到唯一一项 ——
那 B 条来自在线编码器（带梯度，无队列）还是动量编码器（不带梯度）+ 历史队列。

三条路径共用同一个 `supcon_core`、同一个 λ、同一个 τ、同一个分类头、同一份数据、
同样的步数与评估方式（评估**一律纯 CE**，避免 new work 3 里 eval_loss 口径不一致的问题）。

两个容易做错、这里特意处理的点：

1. **EMA 必须跟着 optimizer step，而不是 micro-batch。** grad_accum=8 时
   一个优化步包含 8 次前向；如果每次前向都做一遍完整 EMA，动量系数的实际衰减
   速度就是设定值的 8 倍，而且 bs4(accum8) 与 bs16(accum2) 之间不可比。
   所以 EMA 挂在 `on_optimizer_step` 回调上（transformer 在 `optimizer.step()`
   之后触发它）。**入队**则是每个 micro-batch 都做 —— 队列要的是特征，
   每个 micro-batch 的 key 都是合法特征，没有理由丢掉。
2. **梯度累积不等于大 batch 的对比样本。** 普通 SCL 这一路只能看到当前真实
   micro-batch 内的样本，日志里的 `n_candidates` 会如实反映这一点
   （bs=4 时恒为 4，不随 accum 变化），而 SCL-MoCo 会涨到 4+queue_valid。
3. **普通 SCL 的第二次前向是带梯度的**，所以它比 baseline 慢；这是「加了配对正样本」
   的必然代价，不是实现问题。三组的 CE 都只由视图 1 计算，CE 口径完全一致。
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)
from transformers import Trainer, TrainerCallback

from losses import supcon_core
from moco import MomentumCallbackState

logger = logging.getLogger(__name__)


def build_compute_metrics():
    """Accuracy / Macro-F1 / Macro-P / Macro-R / ROC-AUC（AUC 用概率，不用硬标签）。"""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = np.asarray(logits, dtype=np.float64)
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        preds = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "roc_auc": float(roc_auc_score(labels, probs)),
        }

    return compute_metrics


class MomentumUpdateCallback(TrainerCallback):
    """在每个 optimizer step 之后做一次 EMA。"""

    def __init__(self, branch, momentum: float, state: MomentumCallbackState):
        self.branch = branch
        self.momentum = momentum
        self.mstate = state

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.branch.ema_update(self.momentum)
        self.mstate.ema_calls += 1


class SCLMoCoTrainer(Trainer):
    def __init__(self, *args, bundle=None, method="baseline", lam=0.2,
                 temperature=0.3, momentum=0.999, log_stats_every=50,
                 moco_include_self_key=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.bundle = bundle
        self.method = method
        self.lam = lam
        self.temperature = temperature
        self.momentum = momentum
        self.log_stats_every = log_stats_every
        self.moco_include_self_key = moco_include_self_key
        self.ce_fct = torch.nn.CrossEntropyLoss()
        self.mstate = MomentumCallbackState()
        self.step_log: list[dict] = []
        self._micro_step = 0
        self._view_gap = float("nan")   # 两个视图特征的平均距离，证明视图确实不同
        self._view_cos = float("nan")   # 同一句话在两个编码器下的余弦相似度
        self._contrast_progress = float("nan")   # 对比目标完成度（见 _record_progress）
        self._acc = {"ce": 0.0, "con": 0.0, "total": 0.0, "n": 0,
                     "pos": 0.0, "neg": 0.0, "cand": 0.0, "skipped": 0}
        # 显式关掉「模型自己接管 loss 归一化」这条路径：
        # 打开时 Trainer 不会再除以 gradient_accumulation_steps，
        # 那样 bs4×accum8 与 bs16×accum2 的梯度量级不一致，六组就不可比了。
        self.model_accepts_loss_kwargs = False
        if method == "scl_moco":
            self.add_callback(MomentumUpdateCallback(bundle.momentum, momentum,
                                                     self.mstate))
        logger.info("SCLMoCoTrainer: method=%s λ=%s τ=%s m=%s include_self_key=%s",
                    method, lam, temperature, momentum, moco_include_self_key)

    # ------------------------------------------------------------------ loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels", None)
        if labels is None:
            labels = inputs.pop("label", None)

        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        if labels is None:
            self.bundle.hidden_store.clear()
            return (None, outputs) if return_outputs else None

        ce = self.ce_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        total = ce
        con_value = float("nan")
        stats = None

        # 只在训练时加对比项：评估一律纯 CE，保证 eval_loss 在三组之间可比
        if self.method != "baseline" and model.training:
            # 视图 1 的特征必须在任何第二次前向之前取出来 —— hook 里只留最后一次前向
            feats_v1 = self.bundle.normalized_query_features(inputs["attention_mask"])
            con, stats = self._contrastive_loss(model, inputs, labels, feats_v1)
            total = ce + self.lam * con
            con_value = float(con.detach().item())
            self._record(ce.detach().item(), con_value, float(total.detach().item()),
                         stats)

        if not torch.isfinite(total):
            raise FloatingPointError(
                f"loss 不是有限值：ce={ce.item()} contrast={con_value} "
                f"step={self.state.global_step}")

        self.bundle.hidden_store.clear()
        return (total, outputs) if return_outputs else total

    def _contrastive_loss(self, model, inputs, labels, feats_q):
        """返回 (loss, ContrastiveStats)。feats_q 是视图 1（算 CE 那次前向）的特征。"""
        if self.method == "scl":
            # 普通 SCL：再用**同一个在线编码器**前向一次，dropout 独立重采样得到视图 2。
            # 候选 = 视图 2 的全部 B 条：自身那条是配对正样本，同标签的是正样本，
            # 不同标签的是负样本。不用动量编码器、不用队列。
            # 这样两组的候选结构在队列为空时完全一致，差异只剩「队列 + 动量」。
            model(**inputs)               # hook 里的 hidden state 被第二次前向覆盖
            feats_v2 = self.bundle.normalized_query_features(inputs["attention_mask"])
            self._view_gap = float((feats_q - feats_v2).norm(dim=-1).mean().item())
            self._view_cos = float((feats_q * feats_v2).sum(-1).mean().item())
            loss, stats = supcon_core(feats_q, feats_v2, labels, labels,
                                      self.temperature)
            self._record_progress(loss, stats)
            return loss, stats

        # SCL-MoCo：候选 = 动量编码器给出的本 batch key + 历史队列
        queue = self.bundle.queue
        keys = self.bundle.momentum.encode(inputs)            # [B, d]，no_grad
        cand = torch.cat([keys, queue.feats.to(keys.dtype)], dim=0)
        cand_labels = torch.cat([labels, queue.labels], dim=0)
        usable = torch.cat([torch.ones(keys.shape[0], device=keys.device),
                            queue.usable_mask()], dim=0)
        self_index = None
        if not self.moco_include_self_key:
            self_index = torch.arange(keys.shape[0], device=keys.device)

        self._view_gap = float((feats_q - keys).norm(dim=-1).mean().item())
        # 同一句话在 query / key 两个编码器下的余弦相似度。MoCo 里这一对就是正样本对，
        # 它偏低说明动量编码器太陈旧、模型在对齐一个没意义的目标（m 相对训练步数太大）。
        self._view_cos = float((feats_q * keys).sum(-1).mean().item())
        loss, stats = supcon_core(feats_q, cand, labels, cand_labels,
                                  self.temperature, cand_usable=usable,
                                  self_index=self_index)
        self._record_progress(loss, stats)
        stats.queue_valid = int(queue.valid.item())
        stats.queue_label_counts = queue.label_counts()
        # 算完 loss 再入队：当前 batch 的 key 不应该先进队列再被自己看到两次
        queue.enqueue(keys, labels)
        self.mstate.enqueue_calls += 1
        self.mstate.enqueued_items += int(keys.shape[0])
        return loss, stats

    def _record_progress(self, loss, stats):
        """对比目标的完成度。

        候选数为 N 时，特征完全塌缩（所有候选长得一样）的 loss 恰好是 log(N)；
        完美分开（同类余弦 1、异类 −1）的 loss 有解析下界。实测值落在两者之间的
        位置，才是「对比项到底学没学动」的正确读法 —— 只看 loss 在缓慢下降会误判，
        因为 log(N) 这个常数会把绝对量级顶得很高。
        """
        n_pos = max(stats.pos_per_query, 1e-6)
        n_neg = max(stats.neg_per_query, 0.0)
        n_cand = max(n_pos + n_neg, 1.0)
        tau = self.temperature
        collapsed = math.log(n_cand)
        denom = n_pos * math.exp(1.0 / tau) + n_neg * math.exp(-1.0 / tau)
        floor = -(1.0 / tau - math.log(denom))
        span = collapsed - floor
        value = float(loss.detach().item())
        self._contrast_progress = (collapsed - value) / span if span > 1e-9 else float("nan")

    # ------------------------------------------------------------------ 日志
    def _record(self, ce, con, total, stats):
        acc = self._acc
        acc["ce"] += ce
        acc["con"] += con
        acc["total"] += total
        acc["n"] += 1
        if stats is not None:
            acc["pos"] += stats.pos_per_query
            acc["neg"] += stats.neg_per_query
            acc["cand"] += stats.n_candidates
            acc["skipped"] += stats.n_query - stats.n_query_with_pos
        self._micro_step += 1

        if self._micro_step % self.log_stats_every:
            return
        n = max(acc["n"], 1)
        row = {
            "micro_step": self._micro_step,
            "global_step": self.state.global_step,
            "epoch": round(float(self.state.epoch or 0.0), 4),
            "ce_loss": acc["ce"] / n,
            "contrastive_loss": acc["con"] / n,
            "total_loss": acc["total"] / n,
            "pos_per_query": acc["pos"] / n,
            "neg_per_query": acc["neg"] / n,
            "n_candidates": acc["cand"] / n,
            "queries_skipped_no_pos": acc["skipped"],
            "queue_valid": stats.queue_valid if stats else 0,
            "queue_ptr": int(self.bundle.queue.ptr.item()) if self.bundle.queue else 0,
            "queue_labels": str(stats.queue_label_counts) if stats else "{}",
            "view_gap": self._view_gap,
            "view_cos": self._view_cos,
            "contrast_progress": self._contrast_progress,
            # 该窗口内最后一个 micro-batch 的逐 query 正/负样本数，
            # 用来做「队列为空时两组划分完全一致」的比对
            "pos_counts": str(stats.pos_counts) if stats else "()",
            "neg_counts": str(stats.neg_counts) if stats else "()",
            "ema_calls": self.mstate.ema_calls,
        }
        self.step_log.append(row)
        logger.info(
            "[%s] micro=%d opt_step=%d ep=%.2f | CE %.4f  contrast %.4f  total %.4f | "
            "cand %.0f  pos %.1f  neg %.1f  skipped %d | "
            "q·k %.3f  完成度 %.0f%% | "
            "queue valid=%d ptr=%d %s | ema=%d",
            self.method, row["micro_step"], row["global_step"], row["epoch"],
            row["ce_loss"], row["contrastive_loss"], row["total_loss"],
            row["n_candidates"], row["pos_per_query"], row["neg_per_query"],
            row["queries_skipped_no_pos"], row["view_cos"],
            100 * (row["contrast_progress"] if row["contrast_progress"] ==
                   row["contrast_progress"] else 0), row["queue_valid"],
            row["queue_ptr"], row["queue_labels"], row["ema_calls"])
        for key in acc:
            acc[key] = 0 if isinstance(acc[key], int) else 0.0

    # ---------------------------------------------------------------- 预测
    def predict_in_order(self, dataset):
        """按数据集原始顺序预测。

        `group_by_length=True` 不只作用于训练，也作用于 predict
        （`Trainer._get_eval_sampler` 里会返回 LengthGroupedSampler）。
        不关掉它，概率就会和按文件原序排列的 id 逐行错位 ——
        提交文件行数/格式/分布全都正常，分数却等于随机（原项目实测 AUC 0.5021）。
        """
        previous = self.args.group_by_length
        self.args.group_by_length = False
        try:
            logits = self.predict(dataset).predictions
        finally:
            self.args.group_by_length = previous
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return np.asarray(logits, dtype=np.float64)
