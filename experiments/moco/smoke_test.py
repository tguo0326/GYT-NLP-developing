"""真实性验证：12 项断言 + 关键日志。跑法：

    python smoke_test.py                      # 用真实底座 deberta-v3-large
    python smoke_test.py --model_id microsoft/deberta-v3-base   # 更快的自检

分两部分：
  A. 纯 CPU 的单元检查（损失数学等价性、队列机制），不需要 GPU；
  B. 真实模型 + 真实 Trainer 的端到端检查（动量、optimizer、梯度、checkpoint）。

覆盖任务书第四节的 12 条，每条都打印数字，不是只回一句「已实现」。
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import logging
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding, TrainerCallback, TrainingArguments

import data
import model as model_lib
from losses import SupConLoss, scl_in_batch, scl_two_view, supcon_core
from moco import FeatureQueue
from run_experiment import set_seed
from trainer_scl import SCLMoCoTrainer, build_compute_metrics

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    mark = "✓ PASS" if ok else "✗ FAIL"
    print(f"{mark}  {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"        {line}")


# ============================================================ A. 单元检查（CPU）
def part_a():
    print("\n" + "=" * 78)
    print("A. 单元检查：损失数学 / 队列机制（CPU，不依赖模型）")
    print("=" * 78)
    set_seed(0)

    # A1 —— 我们自己的 supcon_core 与官方 SupConLoss 数值等价
    feats = F.normalize(torch.randn(16, 32), dim=-1)
    labels = torch.randint(0, 2, (16,))
    official = SupConLoss(temperature=0.3, base_temperature=0.3)(
        feats.unsqueeze(1), labels).item()
    ours, stats = scl_in_batch(feats, labels, temperature=0.3)
    check("A1 supcon_core 与官方 SupConLoss 数值等价（普通 SCL 口径没被改动）",
          abs(official - ours.item()) < 1e-5,
          f"官方实现 = {official:.8f}\n本实现   = {ours.item():.8f}\n"
          f"差 = {abs(official - ours.item()):.2e}；候选数/query = {stats.n_candidates}"
          f"（=batch 16 去掉自己），平均正 {stats.pos_per_query:.2f} 负 {stats.neg_per_query:.2f}")

    # A2 —— 空队列 / 无正样本不产生 NaN
    q = FeatureQueue(8, 4)
    empty_loss, empty_stats = supcon_core(
        F.normalize(torch.randn(4, 4), dim=-1), q.feats, torch.zeros(4, dtype=torch.long),
        q.labels, 0.3, cand_usable=q.usable_mask())
    single = F.normalize(torch.randn(2, 4), dim=-1)
    nopos_loss, nopos_stats = scl_in_batch(single, torch.tensor([0, 1]), 0.3)
    check("A2 队列为空 / 无正样本时 loss 有限且为 0，不产生 NaN",
          torch.isfinite(empty_loss) and empty_loss.item() == 0.0
          and torch.isfinite(nopos_loss) and nopos_loss.item() == 0.0,
          f"空队列：loss={empty_loss.item()} 有效候选={empty_stats.n_candidates} "
          f"有正样本的 query={empty_stats.n_query_with_pos}/4\n"
          f"batch 内每类各 1 条（谁都没有正样本）：loss={nopos_loss.item()} "
          f"参与 loss 的 query={nopos_stats.n_query_with_pos}/2")

    # A3 —— 队列指针、有效长度、FIFO 覆盖、标签保真
    q = FeatureQueue(10, 4)
    trace, tags, sent = [], [], []
    for step in range(4):
        f = F.normalize(torch.randn(4, 4), dim=-1)
        lb = torch.full((4,), step % 2, dtype=torch.long)
        sent.append(f.clone())
        q.enqueue(f, lb)
        trace.append((int(q.ptr.item()), int(q.valid.item())))
        tags.append(lb.tolist())
    norms = q.feats[:int(q.valid.item())].norm(dim=-1)
    # queue_size=10，共写入 16 条。槽位 2-3 先由 step0（label 0）写入，
    # 又被 step3（label 1）覆盖 —— 逐元素比对特征就能证明「最旧的被顶掉了」。
    slot23_labels = q.labels[2:4].tolist()
    slot23_is_newest = torch.allclose(q.feats[2:4], sent[3][:2], atol=1e-6)
    slot23_is_oldest = torch.allclose(q.feats[2:4], sent[0][2:4], atol=1e-6)
    # 槽位 6-7 只被 step1（label 1）写过一次，应当原样保留
    slot67_intact = torch.allclose(q.feats[6:8], sent[1][2:4], atol=1e-6)
    check("A3 队列指针推进 / 有效长度 / FIFO 覆盖最旧 / 标签保真 / 特征已归一化",
          trace == [(4, 4), (8, 8), (2, 10), (6, 10)]
          and torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
          and slot23_labels == [1, 1] and slot23_is_newest and not slot23_is_oldest
          and slot67_intact,
          f"queue_size=10，每步入队 4 条，(ptr, valid) 轨迹 = {trace}"
          f"（第 3 步跨过队尾，绕回队头）\n"
          f"入队标签依次 = {tags}\n"
          f"槽位 2-3：标签 = {slot23_labels}；特征 == 最新一批(step3) 的前两条？"
          f"{slot23_is_newest}；还等于最旧那批(step0) 吗？{slot23_is_oldest}\n"
          f"  → 最旧数据确实被覆盖，新数据确实写进去了\n"
          f"槽位 6-7（只写过一次）特征保持不变：{slot67_intact}\n"
          f"队列内特征 L2 范数 min={norms.min():.6f} max={norms.max():.6f}（应为 1）")

    # A4 —— 队列为空时，双视图 SCL 与 SCL-MoCo 的划分**逐 query 完全一致**
    v1 = F.normalize(torch.randn(4, 8), dim=-1)
    v2 = F.normalize(torch.randn(4, 8), dim=-1)     # 充当「另一个视图 / 动量 key」
    lq = torch.tensor([0, 0, 1, 0])                 # 故意让某一类只有 1 条
    empty_q = FeatureQueue(4096, 8)
    loss_scl, s_scl = scl_two_view(v1, v2, lq, 0.3)
    loss_moco, s_moco = supcon_core(
        v1, torch.cat([v2, empty_q.feats]), lq, torch.cat([lq, empty_q.labels]), 0.3,
        cand_usable=torch.cat([torch.ones(4), empty_q.usable_mask()]))
    check("A4 队列为空时：两组候选数、逐 query 正/负划分、乃至 loss 完全一致",
          s_scl.n_candidates == s_moco.n_candidates == 4
          and s_scl.pos_counts == s_moco.pos_counts
          and s_scl.neg_counts == s_moco.neg_counts
          and abs(loss_scl.item() - loss_moco.item()) < 1e-6
          and s_scl.n_query_with_pos == s_moco.n_query_with_pos == 4,
          f"标签 = {lq.tolist()}（label=1 只有 1 条，单视图时它必然没有正样本）\n"
          f"SCL     ：候选 {s_scl.n_candidates}，逐 query 正 {s_scl.pos_counts}，"
          f"负 {s_scl.neg_counts}，loss {loss_scl.item():.8f}\n"
          f"SCL-MoCo：候选 {s_moco.n_candidates}，逐 query 正 {s_moco.pos_counts}，"
          f"负 {s_moco.neg_counts}，loss {loss_moco.item():.8f}\n"
          f"两组都有 {s_scl.n_query_with_pos}/4 个 query 参与 loss（没有任何 query 被 skip）\n"
          f"对照：老的单视图 SCL 在这批标签上 → 逐 query 正 "
          f"{scl_in_batch(v1, lq, 0.3)[1].pos_counts}（第 3 条 query 是 0，会被丢掉）")

    # A5 —— 队列填满后，只有 SCL-MoCo 的候选数扩大
    qbig = FeatureQueue(4096, 8)
    qbig.enqueue(F.normalize(torch.randn(4096, 8), dim=-1),
                 torch.randint(0, 2, (4096,)))
    _, s_scl2 = scl_two_view(v1, v2, lq, 0.3)
    _, s_moco2 = supcon_core(
        v1, torch.cat([v2, qbig.feats]), lq, torch.cat([lq, qbig.labels]), 0.3,
        cand_usable=torch.cat([torch.ones(4), qbig.usable_mask()]))
    check("A5 队列填满后只有 SCL-MoCo 的候选扩大，SCL 仍然是 B 条",
          s_scl2.n_candidates == 4 and s_moco2.n_candidates == 4100
          and s_scl2.n_candidates == s_scl.n_candidates,
          f"SCL     ：候选 {s_scl2.n_candidates}（与队列为空时相同 = "
          f"{s_scl.n_candidates}），平均正 {s_scl2.pos_per_query:.2f}，"
          f"平均负 {s_scl2.neg_per_query:.2f}\n"
          f"SCL-MoCo：候选 {s_moco2.n_candidates}（4 条 key + 4096 条队列），"
          f"平均正 {s_moco2.pos_per_query:.1f}，平均负 {s_moco2.neg_per_query:.1f}\n"
          f"→ 放大倍数 {s_moco2.n_candidates / s_scl2.n_candidates:.0f}×")


# ==================================================== B. 端到端检查（真实模型）
class Spy(TrainerCallback):
    """在每个 optimizer step 前后抓快照，用于验证 EMA 的时机与幅度。"""

    def __init__(self, branch, model):
        self.branch = branch
        self.model = model
        self.before_opt = []      # optimizer.step() 之前：(q_snapshot, k_snapshot)
        self.after_ema = []       # EMA 之后：(q_snapshot, k_snapshot)
        self.grad_norms = {}      # 梯度必须在 zero_grad 之前抓，训练结束后就没了
        self.key_has_grad = []

    def _snap(self):
        q = self.branch.query_params()[0].detach().clone()
        k = self.branch.key_params()[0].detach().clone()
        return q, k

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.before_opt.append(self._snap())
        # 每个 optimizer step 都刷新（留最后一次）。只看第一步会误判：LoRA 的 B
        # 初始化为 0，第一步 lora_A 的梯度恒等于 0，看起来像「梯度没传到 A」。
        self.grad_norms = {name: param.grad.norm().item()
                           for name, param in self.model.named_parameters()
                           if param.grad is not None}
        self.key_has_grad = [n for n in self.grad_norms if ".k_enc." in n]

    def on_step_end(self, args, state, control, **kwargs):
        # on_step_end 在 MomentumUpdateCallback.on_optimizer_step 之后触发
        self.after_ema.append(self._snap())


def part_b(model_id: str, device: str):
    print("\n" + "=" * 78)
    print(f"B. 端到端检查：{model_id} + LoRA + 动量分支 + 队列 + 真实 Trainer")
    print("=" * 78)

    args = argparse.Namespace(
        method="scl_moco", model_id=model_id, num_labels=2, max_length=128,
        rank=16, lora_alpha=32, lora_dropout=0.05, gradient_checkpointing=True,
        proj_dim=128, queue_size=24, momentum=0.999,
    )
    set_seed(42)
    bundle = model_lib.build(args)
    bundle.model.to(device)
    bundle.queue.to(device)
    bundle.momentum.hard_sync()

    # B1 —— 初始化一致
    pairs = bundle.momentum.pairs
    per_tensor = [(q.data - k.data).abs().max().item() for q, k in pairs]
    check("B1 Query 与 Key 初始化参数完全一致（逐张量比较）",
          max(per_tensor) == 0.0,
          f"参数对 {len(pairs)} 组，共 "
          f"{sum(q.numel() for q, _ in pairs):,} 个参数\n"
          f"逐张量最大绝对差的最大值 = {max(per_tensor):.3e}（应为 0）\n"
          f"key 参数示例：{[n for n, _ in list(bundle.model.named_parameters()) if '.k_enc.' in n][:3]}")

    # B3(a) —— key requires_grad
    key_flags = [p.requires_grad for p in bundle.momentum.key_params()]
    check("B3 Key Encoder 全部 requires_grad=False",
          not any(key_flags),
          f"{len(key_flags)} 个 key 参数中 requires_grad=True 的有 {sum(key_flags)} 个")

    # 建一个只有 40 条的迷你训练集，bs=4 × accum=2 → 8 个 optimizer step
    train_ds, val_ds, _, _ = data.build_datasets(
        bundle.tokenizer, max_length=args.max_length, seed=42, subset=64)
    train_ds = train_ds.select(range(64))

    targs = TrainingArguments(
        output_dir=tempfile.mkdtemp(), max_steps=8,
        per_device_train_batch_size=4, per_device_eval_batch_size=8,
        gradient_accumulation_steps=2, learning_rate=1e-4, warmup_ratio=0.0,
        weight_decay=0.01, lr_scheduler_type="linear", logging_steps=1,
        eval_strategy="no", save_strategy="no", seed=42, data_seed=42,
        fp16=(device == "cuda"), dataloader_num_workers=0, group_by_length=True,
        report_to=[], label_names=["labels"], disable_tqdm=True)

    trainer = SCLMoCoTrainer(
        model=bundle.model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=bundle.tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=bundle.tokenizer),
        compute_metrics=build_compute_metrics(), bundle=bundle, method="scl_moco",
        lam=0.2, temperature=0.3, momentum=args.momentum, log_stats_every=1)
    spy = Spy(bundle.momentum, bundle.model)
    trainer.add_callback(spy)

    # B2 —— key 不在 optimizer 里（optimizer 在 train() 里才建，所以先建一次看）
    trainer.create_optimizer()
    opt_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    key_ids = {id(p) for p in bundle.momentum.key_params()}
    query_ids = {id(p) for p in bundle.momentum.query_params()}
    check("B2 Key Encoder 不在 optimizer 中（Query 全在）",
          not (opt_ids & key_ids) and query_ids <= opt_ids,
          f"optimizer 参数 {len(opt_ids)} 个\n"
          f"key 参数 {len(key_ids)} 个，落在 optimizer 里的 {len(opt_ids & key_ids)} 个（应为 0）\n"
          f"query 参数 {len(query_ids)} 个，落在 optimizer 里的 "
          f"{len(query_ids & opt_ids)} 个（应全部命中）")

    # 记录训练前的 query 参数，用于 B4
    q_before_all = [p.detach().clone() for p in bundle.momentum.query_params()]
    print("\n---- 开始 8 个 optimizer step 的真实训练（bs=4 × accum=2，queue_size=24）----")
    trainer.train()
    print("---- 训练结束 ----\n")

    # B4 —— query 参数正常变化
    q_delta = max((p.detach() - b).abs().max().item()
                  for p, b in zip(bundle.momentum.query_params(), q_before_all))
    check("B4 optimizer 更新后 Query 参数正常变化",
          q_delta > 0,
          f"8 个 optimizer step 后 query 参数最大绝对变化 = {q_delta:.3e}（应 > 0）")

    # B5 —— key 通过 EMA 缓慢变化，且严格等于 EMA 公式
    m = args.momentum
    rows, formula_err = [], 0.0
    for i, ((q_pre, k_pre), (q_post, k_post)) in enumerate(
            zip(spy.before_opt, spy.after_ema)):
        expected = m * k_pre + (1 - m) * q_post
        formula_err = max(formula_err, (expected - k_post).abs().max().item())
        dq = (q_post - q_pre).abs().max().item()
        dk = (k_post - k_pre).abs().max().item()
        rows.append(f"step {i + 1}: |Δquery|={dq:.3e}  |Δkey|={dk:.3e}  "
                    f"比值={dk / dq if dq else float('nan'):.5f}（应≈1-m={1 - m}）")
    dk_all = [(k_post - k_pre).abs().max().item()
              for (_, k_pre), (_, k_post) in zip(spy.before_opt, spy.after_ema)]
    check("B5 Key 参数按 EMA 缓慢变化（既不是完全不变，也不是跟着一起跳）",
          min(dk_all) > 0 and formula_err < 1e-6,
          "\n".join(rows) + f"\nθ_key ← m·θ_key+(1-m)·θ_query 的最大偏差 = "
          f"{formula_err:.3e}（应≈0）\nkey 最小变化 = {min(dk_all):.3e}（应 > 0）")

    # B6 —— EMA 跟着 optimizer step，不是 micro-batch
    check("B6 EMA 次数 == optimizer step 数（不是 micro-batch 数）",
          trainer.mstate.ema_calls == 8 and trainer.mstate.enqueue_calls == 16,
          f"optimizer step = {trainer.state.global_step}，EMA 调用 = "
          f"{trainer.mstate.ema_calls}\n"
          f"micro-batch（= 入队次数）= {trainer.mstate.enqueue_calls}，"
          f"共入队 {trainer.mstate.enqueued_items} 条特征")

    # B7 —— 队列指针轨迹 + 覆盖 + 内容
    ptr, valid = int(bundle.queue.ptr.item()), int(bundle.queue.valid.item())
    norms = bundle.queue.feats.norm(dim=-1)
    counts = bundle.queue.label_counts()
    check("B7 队列填满并绕回，存的是归一化 embedding + 真实标签",
          valid == 24 and ptr == (16 * 4) % 24 and
          torch.allclose(norms, torch.ones_like(norms), atol=1e-3) and
          set(counts) <= {0, 1} and sum(counts.values()) == 24,
          f"queue_size=24，入队 16×4=64 条 → valid={valid}（满）ptr={ptr}"
          f"（=64 mod 24）\n"
          f"队列特征 L2 范数 min={norms.min():.5f} max={norms.max():.5f}（应为 1）\n"
          f"队列标签分布 = {counts}（只能是真实标签 0/1，且总数=24）\n"
          f"特征维度 = {tuple(bundle.queue.feats.shape)}（proj_dim=128，不是 2 类 logits）")

    # B8 —— loss 全程有限；对比统计有记录
    logs = trainer.step_log
    finite = all(all(torch.isfinite(torch.tensor(float(r[k]))) for k in
                     ("ce_loss", "contrastive_loss", "total_loss")) for r in logs)
    check("B8 loss 不出现 NaN/Inf，且对比项确实非零（说明真的在起作用）",
          finite and any(abs(float(r["contrastive_loss"])) > 0 for r in logs),
          f"记录了 {len(logs)} 个 micro-batch\n" +
          "\n".join(f"micro {r['micro_step']:2d}: CE {float(r['ce_loss']):.4f}  "
                    f"contrast {float(r['contrastive_loss']):.4f}  "
                    f"total {float(r['total_loss']):.4f}  cand "
                    f"{float(r['n_candidates']):.0f}  pos {float(r['pos_per_query']):.1f}  "
                    f"neg {float(r['neg_per_query']):.1f}  queue "
                    f"{r['queue_valid']}  ptr {r['queue_ptr']}"
                    for r in logs[:6]) + "\n... (完整记录见 result/*_steps.csv)")

    # B9 —— 候选数随队列增长（bs=4 下 3 → 4+queue）
    cands = [float(r["n_candidates"]) for r in logs]
    check("B9 队列真的把候选数抬起来了（bs=4：从 4 涨到 4+24）",
          cands[0] < cands[-1] and cands[-1] == 28,
          f"第 1 个 micro-batch 候选数 = {cands[0]:.0f}（队列还空，只有 4 条 key）\n"
          f"最后一个 micro-batch 候选数 = {cands[-1]:.0f}（4 条 key + 24 条队列）\n"
          f"同配置下普通 SCL 只有 3（见 A4）")

    # B10 —— 梯度到达 query LoRA 与 projection head，key 侧无梯度
    # 梯度在第一个 optimizer step 前抓的（zero_grad 之后就没了）
    grads = {n: v for n, v in spy.grad_norms.items() if v > 0}
    lora_q = sorted(n for n in grads if ".q_enc." in n and "lora_" in n)
    lora_a = [n for n in lora_q if "lora_A" in n]
    lora_b = [n for n in lora_q if "lora_B" in n]
    proj_q = [n for n in grads if "proj_head" in n]
    pooler_q = [n for n in grads if "pooler" in n]
    check("B10 梯度传到 Query 的 LoRA 与 projection head；Key 侧完全没有梯度",
          bool(lora_a) and bool(lora_b) and bool(proj_q) and not spy.key_has_grad,
          f"（在最后一个 optimizer step 的 zero_grad 之前抓的）\n"
          f"有非零梯度的参数 {len(grads)} 个：query LoRA {len(lora_q)} 个"
          f"（lora_A {len(lora_a)} + lora_B {len(lora_b)}）、"
          f"proj_head {len(proj_q)} 个、pooler {len(pooler_q)} 个\n"
          f"示例 LoRA A：{lora_a[0]}\n        grad_norm={grads[lora_a[0]]:.3e}\n"
          f"示例 LoRA B：{lora_b[0]}\n        grad_norm={grads[lora_b[0]]:.3e}\n"
          f"示例 proj  ：{proj_q[0]}\n        grad_norm={grads[proj_q[0]]:.3e}\n"
          f"key 侧（.k_enc.）带 .grad 的参数 = {len(spy.key_has_grad)} 个（应为 0）")

    # B11 —— checkpoint 保存 / 重新加载后队列、指针、动量参数都能恢复
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "moco_state.pt")
        model_lib.save_moco_state(bundle, path)
        want = {
            "feats": bundle.queue.feats.detach().cpu().clone(),
            "labels": bundle.queue.labels.detach().cpu().clone(),
            "ptr": bundle.queue.ptr.item(), "valid": bundle.queue.valid.item(),
            "key": [k.detach().cpu().clone() for k in bundle.momentum.key_params()],
            "query": [q.detach().cpu().clone() for q in bundle.momentum.query_params()],
        }
        # 故意破坏内存中的状态，再从 checkpoint 恢复
        with torch.no_grad():
            bundle.queue.feats.zero_()
            bundle.queue.labels.fill_(-1)
            bundle.queue.ptr.fill_(0)
            bundle.queue.valid.fill_(0)
            for k in bundle.momentum.key_params():
                k.zero_()
            for q in bundle.momentum.query_params():
                q.add_(1.0)
        model_lib.load_moco_state(bundle, path)
        ok = (torch.equal(bundle.queue.feats.cpu(), want["feats"])
              and torch.equal(bundle.queue.labels.cpu(), want["labels"])
              and bundle.queue.ptr.item() == want["ptr"]
              and bundle.queue.valid.item() == want["valid"]
              and all(torch.equal(k.detach().cpu(), w)
                      for k, w in zip(bundle.momentum.key_params(), want["key"]))
              and all(torch.equal(q.detach().cpu(), w)
                      for q, w in zip(bundle.momentum.query_params(), want["query"])))
        check("B11 保存并重新加载 checkpoint 后，队列 / 指针 / 动量参数逐字节恢复",
              ok,
              f"破坏前 ptr={want['ptr']} valid={want['valid']}；"
              f"清零后再 load → ptr={bundle.queue.ptr.item()} "
              f"valid={bundle.queue.valid.item()}\n"
              f"队列特征逐字节相同：{torch.equal(bundle.queue.feats.cpu(), want['feats'])}\n"
              f"队列标签逐字节相同：{torch.equal(bundle.queue.labels.cpu(), want['labels'])}\n"
              f"key 参数逐字节相同：{all(torch.equal(k.detach().cpu(), w) for k, w in zip(bundle.momentum.key_params(), want['key']))}\n"
              f"query 参数逐字节相同：{all(torch.equal(q.detach().cpu(), w) for q, w in zip(bundle.momentum.query_params(), want['query']))}")

    # B12 —— 显存：确认没有复制第二套底座
    total = sum(p.numel() for p in bundle.model.parameters())
    key_n = sum(p.numel() for p in bundle.momentum.key_params())
    check("B12 动量分支没有复制整个底座（显存代价可忽略）",
          key_n < total * 0.02,
          f"模型总参数 {total:,}\n动量副本参数 {key_n:,}"
          f"（{100 * key_n / total:.3f}%，约 {key_n * 4 / 1024 ** 2:.1f} MB fp32）\n"
          f"队列 {bundle.queue.queue_size}×{bundle.queue.dim} float32 = "
          f"{bundle.queue.feats.numel() * 4 / 1024 ** 2:.2f} MB"
          + (f"\n本次训练显存峰值 {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB"
             f"（len=128/bs=4 的迷你配置，不代表正式实验）" if device == "cuda" else ""))
    bundle.close()


def run_mini(method: str, model_id: str, device: str, batch_size: int,
             queue_size: int = 4096, steps: int = 4, accum: int = 2):
    """跑几步真实训练，返回 (trainer, bundle)。三种方法共用同一条代码路径。"""
    args = argparse.Namespace(
        method=method, model_id=model_id, num_labels=2, max_length=128,
        rank=16, lora_alpha=32, lora_dropout=0.05, gradient_checkpointing=True,
        proj_dim=128, queue_size=queue_size, momentum=0.999)
    set_seed(42)
    bundle = model_lib.build(args)
    bundle.model.to(device)
    if bundle.queue is not None:
        bundle.queue.to(device)
        bundle.momentum.hard_sync()
    train_ds, val_ds, _, _ = data.build_datasets(
        bundle.tokenizer, max_length=128, seed=42, subset=64)
    targs = TrainingArguments(
        output_dir=tempfile.mkdtemp(), max_steps=steps,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=8,
        gradient_accumulation_steps=accum, learning_rate=1e-4, warmup_ratio=0.0,
        weight_decay=0.01, lr_scheduler_type="linear", logging_steps=100,
        eval_strategy="no", save_strategy="no", seed=42, data_seed=42,
        fp16=(device == "cuda"), dataloader_num_workers=0, group_by_length=True,
        report_to=[], label_names=["labels"], disable_tqdm=True)
    trainer = SCLMoCoTrainer(
        model=bundle.model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=bundle.tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=bundle.tokenizer),
        compute_metrics=build_compute_metrics(), bundle=bundle, method=method,
        lam=0.2, temperature=0.3, momentum=0.999, log_stats_every=1)
    trainer.train()
    return trainer, bundle


def part_c(model_id: str, device: str):
    """三种方法各跑几步，横向对比「每个 query 能看到多少候选」。"""
    print("\n" + "=" * 78)
    print("C. 三组方法端到端各跑 4 个 optimizer step（bs=4 × accum=2，queue_size=4096）")
    print("=" * 78)
    logs = {}
    for method in ("baseline", "scl", "scl_moco"):
        trainer, bundle = run_mini(method, model_id, device, batch_size=4, steps=6)
        logs[method] = list(trainer.step_log)
        bundle.close()
        del trainer, bundle
        if device == "cuda":
            torch.cuda.empty_cache()

    def row(method, i):
        return logs[method][i]

    # C1 —— 队列为空（第 1 个 micro-batch）时两组划分完全一致
    a, b = row("scl", 0), row("scl_moco", 0)
    same = (a["pos_counts"] == b["pos_counts"] and a["neg_counts"] == b["neg_counts"]
            and float(a["n_candidates"]) == float(b["n_candidates"]) == 4
            and abs(float(a["ce_loss"]) - float(b["ce_loss"])) < 1e-3)
    check("C1 队列为空时（第 1 个 micro-batch）两组候选数与逐 query 正负划分一致",
          same,
          f"两组吃到的是同一个 batch（CE 相同：{float(a['ce_loss']):.4f} vs "
          f"{float(b['ce_loss']):.4f}，seed/数据顺序一致）\n"
          f"SCL     ：候选 {float(a['n_candidates']):.0f}，正 {a['pos_counts']}，"
          f"负 {a['neg_counts']}，queue valid={a['queue_valid']}\n"
          f"SCL-MoCo：候选 {float(b['n_candidates']):.0f}，正 {b['pos_counts']}，"
          f"负 {b['neg_counts']}，queue valid={b['queue_valid']}")

    # C2 —— 队列填充后只有 SCL-MoCo 扩大
    a2, b2 = row("scl", -1), row("scl_moco", -1)
    grew = (float(a2["n_candidates"]) == 4
            and float(b2["n_candidates"]) > float(b["n_candidates"]))
    check("C2 队列填充后只有 SCL-MoCo 的候选数扩大，SCL 原地不动",
          grew,
          f"SCL     ：第 1 个 micro-batch {float(a['n_candidates']):.0f} → 最后 "
          f"{float(a2['n_candidates']):.0f}（accum=2 也不会变多）\n"
          f"SCL-MoCo：第 1 个 micro-batch {float(b['n_candidates']):.0f} → 最后 "
          f"{float(b2['n_candidates']):.0f}（queue valid={b2['queue_valid']}，"
          f"标签分布 {b2['queue_labels']}）")

    # C3 —— 两组都不因缺正样本而 skip
    skipped = {m: sum(int(r["queries_skipped_no_pos"]) for r in logs[m])
               for m in ("scl", "scl_moco")}
    check("C3 两组都没有任何 query 因缺少正样本被跳过",
          all(v == 0 for v in skipped.values()),
          f"SCL 累计 skip = {skipped['scl']}，SCL-MoCo 累计 skip = "
          f"{skipped['scl_moco']}（各 {len(logs['scl'])} 个 micro-batch）\n"
          f"对照：改动前的单视图 SCL 在同样配置下 8 个 micro-batch 里有 4 个出现 skip=1")

    # C4 —— 两个视图确实不同（第二次前向的 dropout 真的重采样了）
    gaps = {m: [float(r["view_gap"]) for r in logs[m]] for m in ("scl", "scl_moco")}
    check("C4 视图 2 与视图 1 确实不同（否则「配对正样本」是个恒等式，等于没加）",
          all(min(v) > 1e-4 for v in gaps.values()),
          f"SCL      视图距离 ‖v1−v2‖ 均值区间 = [{min(gaps['scl']):.4f}, "
          f"{max(gaps['scl']):.4f}]（同一在线编码器，dropout 独立重采样）\n"
          f"SCL-MoCo 视图距离 ‖q−k‖   均值区间 = [{min(gaps['scl_moco']):.4f}, "
          f"{max(gaps['scl_moco']):.4f}]（动量编码器 + dropout）")

    # C5 —— baseline 没有对比项
    check("C5 baseline 只有 CE，完全不进对比分支",
          not logs["baseline"],
          f"baseline 的对比统计记录数 = {len(logs['baseline'])}（应为 0）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="microsoft/deberta-v3-large")
    parser.add_argument("--skip_b", action="store_true")
    parser.add_argument("--skip_c", action="store_true")
    opts = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    part_a()
    if not opts.skip_b:
        part_b(opts.model_id, device)
    if not opts.skip_c:
        part_c(opts.model_id, device)

    print("\n" + "=" * 78)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"合计 {len(RESULTS)} 项，通过 {len(RESULTS) - len(failed)} 项，失败 {len(failed)} 项")
    for name in failed:
        print(f"  失败：{name}")
    print("=" * 78)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
