"""一组实验的入口。配置默认值 100% 对齐 0.9633 那次 DeBERTa-v3-large + LoRA。

    python run_experiment.py --method baseline --batch_size 4
    python run_experiment.py --method scl      --batch_size 4
    python run_experiment.py --method scl_moco --batch_size 4

除 `--method` 和 `--batch_size` 外不要动其他参数：三组之间只允许差方法本身，
`--grad_accum` 默认自动取 `32 // batch_size`，使 effective batch 恒为 32
（与 0.9633 那次的 effective batch 一致）。
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# PEFT 训练里 activation 形状随 batch 内最长样本变化，固定 block 复用率低
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)
from transformers import DataCollatorWithPadding, TrainerCallback, TrainingArguments

import data
import model as model_lib
from trainer_scl import SCLMoCoTrainer, build_compute_metrics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
EFFECTIVE_BATCH = 32          # 与 NLP/core/peft_trainer.py 的 EFFECTIVE_BATCH 一致

logger = logging.getLogger("run_experiment")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=model_lib.METHODS)
    p.add_argument("--model_id", default="microsoft/deberta-v3-large")
    p.add_argument("--num_labels", type=int, default=2)
    p.add_argument("--data_dir", default=data.DEFAULT_DATA_DIR)

    # ↓↓↓ 以下默认值全部来自 NLP/results/deberta_lora_summary.json（0.9633 那次）
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=0,
                   help="0 = 自动取 32 // batch_size，保证 effective batch 恒为 32")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--seed", type=int, default=data.SEED)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())

    # ↓↓↓ 对比学习：λ 与 τ 沿用 new work 3 的 SCL 实验（α=0.2, τ=0.3）
    p.add_argument("--lam", type=float, default=0.2, help="对比损失权重 λ")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--proj_dim", type=int, default=128)
    # ↓↓↓ MoCo
    p.add_argument("--queue_size", type=int, default=4096)
    p.add_argument("--momentum", type=float, default=0.999)
    p.add_argument("--moco_include_self_key", action="store_true", default=True,
                   help="把 query 自己的动量 key 当正样本（原始 MoCo 做法，默认开）")
    p.add_argument("--moco_exclude_self_key", dest="moco_include_self_key",
                   action="store_false")

    p.add_argument("--log_stats_every", type=int, default=50,
                   help="每多少个 micro-batch 打一条对比统计日志")
    p.add_argument("--subset", type=int, default=0, help="仅冒烟测试用")
    p.add_argument("--probe_steps", type=int, default=0,
                   help="只跑这么多 optimizer step 测速度和显存峰值就退出")
    p.add_argument("--tag", default=None)
    p.add_argument("--out_dir", default=str(REPO / "results"))
    p.add_argument("--log_dir", default=str(REPO / "logs"))
    p.add_argument("--ckpt_dir", default=str(REPO / "models"))
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(tag: str, log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(Path(log_dir) / f"{tag}.log", mode="w",
                                     encoding="utf-8"),
                  logging.StreamHandler()])
    logging.getLogger("run_experiment").info("命令：%s", " ".join(sys.argv))


class LogToFileCallback(TrainerCallback):
    """把 Trainer 的训练日志也走一遍 logging，写进 logs/<tag>.log。

    Trainer 自己那条 `{'loss': ...}` 是 `PrinterCallback` 用 print 打的，
    stdout 重定向到文件时会被缓冲，几十分钟看不到一行 —— 进度没法追。
    这个回调只写日志，不改任何训练数值。
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        parts = " ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in logs.items())
        logger.info("[trainer] step %d/%s %s", state.global_step,
                    state.max_steps or "?", parts)


class SanityCallback(TrainerCallback):
    """训练真正开始时（optimizer 已建好）做几条硬断言，写进日志留证。"""

    def __init__(self, bundle):
        self.bundle = bundle
        self.checked = False

    def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
        if self.checked or self.bundle.momentum is None:
            return
        self.checked = True
        opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        key_ids = {id(p) for p in self.bundle.momentum.key_params()}
        query_ids = {id(p) for p in self.bundle.momentum.query_params()}
        overlap = opt_ids & key_ids
        logger.info("[断言] optimizer 参数 %d 个；key 参数 %d 个，其中在 optimizer 里的 %d 个",
                    len(opt_ids), len(key_ids), len(overlap))
        assert not overlap, "key 参数进了 optimizer！"
        assert query_ids <= opt_ids, "有 query 参数没进 optimizer！"
        assert all(not p.requires_grad for p in self.bundle.momentum.key_params()), \
            "key 参数的 requires_grad 不是 False"
        diff = self.bundle.momentum.max_abs_diff()
        logger.info("[断言] 训练开始时 query 与 key 参数最大绝对差 = %.3e（应为 0）", diff)
        assert diff == 0.0, "初始化时 query 与 key 不一致"


def full_metrics(y_true, probs) -> dict:
    preds = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "macro_f1": round(float(f1), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
    }


def main(argv=None):
    args = parse_args(argv)
    accum = args.grad_accum or max(1, EFFECTIVE_BATCH // max(args.batch_size, 1))
    tag = args.tag or f"scl_moco_{args.method}_bs{args.batch_size}_seed{args.seed}"
    setup_logging(tag, args.log_dir)
    set_seed(args.seed)

    logger.info("method=%s batch=%d × accum=%d（effective %d）max_length=%d lr=%g "
                "epochs=%s seed=%d λ=%s τ=%s queue=%d m=%s",
                args.method, args.batch_size, accum, args.batch_size * accum,
                args.max_length, args.lr, args.epochs, args.seed, args.lam,
                args.temperature, args.queue_size, args.momentum)

    bundle = model_lib.build(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle.model.to(device)
    if bundle.queue is not None:
        bundle.queue.to(device)
        bundle.momentum.hard_sync()          # .to() 之后再同步一次，确保严格一致

    train_ds, val_ds, test_ds, test_frame = data.build_datasets(
        bundle.tokenizer, data_dir=args.data_dir, max_length=args.max_length,
        seed=args.seed, subset=args.subset)

    training_args = TrainingArguments(
        output_dir=str(Path(args.ckpt_dir) / f"{tag}_hf"),
        num_train_epochs=args.epochs,
        max_steps=args.probe_steps or -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 8),
        gradient_accumulation_steps=accum,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        logging_dir=str(Path(args.log_dir) / tag),
        logging_steps=50,
        eval_strategy="no" if args.probe_steps else "epoch",
        save_strategy="no",
        seed=args.seed,
        data_seed=args.seed,          # 数据顺序只由 seed 决定，与方法无关
        fp16=args.fp16,
        dataloader_num_workers=0,
        group_by_length=True,
        report_to=[],
        label_names=["labels"],
        disable_tqdm=not sys.stderr.isatty(),
    )

    trainer = SCLMoCoTrainer(
        model=bundle.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=bundle.tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=bundle.tokenizer),
        compute_metrics=build_compute_metrics(),
        bundle=bundle,
        method=args.method,
        lam=args.lam,
        temperature=args.temperature,
        momentum=args.momentum,
        log_stats_every=args.log_stats_every,
        moco_include_self_key=args.moco_include_self_key,
    )
    trainer.add_callback(SanityCallback(bundle))
    trainer.add_callback(LogToFileCallback())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer.train()
    train_seconds = time.time() - started
    peak_gb = (torch.cuda.max_memory_allocated() / 1024 ** 3
               if torch.cuda.is_available() else 0.0)
    logger.info("训练结束：%.1fs，显存峰值 %.2f GB（reserved %.2f GB）", train_seconds,
                peak_gb,
                torch.cuda.max_memory_reserved() / 1024 ** 3
                if torch.cuda.is_available() else 0.0)

    if args.method == "scl_moco":
        logger.info("EMA 调用 %d 次；入队 %d 次共 %d 条；队列有效长度 %d，指针 %d，标签分布 %s",
                    trainer.mstate.ema_calls, trainer.mstate.enqueue_calls,
                    trainer.mstate.enqueued_items, int(bundle.queue.valid.item()),
                    int(bundle.queue.ptr.item()), bundle.queue.label_counts())
        expected = trainer.state.global_step
        assert trainer.mstate.ema_calls == expected, \
            f"EMA 次数 {trainer.mstate.ema_calls} != optimizer step 数 {expected}"
        logger.info("[断言] EMA 次数 == optimizer step 数 == %d（不是 micro-batch 数 %d）",
                    expected, trainer.mstate.enqueue_calls)

    if args.probe_steps:
        logger.info("探测完成：%d step，%.2fs/step，峰值 %.2f GB",
                    args.probe_steps, train_seconds / args.probe_steps, peak_gb)
        return {"probe": True, "seconds_per_step": train_seconds / args.probe_steps,
                "peak_gpu_gb": round(peak_gb, 2)}

    # ---- 验证集 ----
    val_metrics = trainer.evaluate()
    logger.info("验证集：%s", val_metrics)

    # ---- 测试集（进程内推理，绝不事后重载 adapter：pooler 随机初始化不可复现）----
    logits = trainer.predict_in_order(test_ds)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data.save_submission(test_frame, probs, str(out_dir / f"{tag}_submission.csv"))

    test_metrics = None
    truth = data.load_test_labels(test_frame, args.data_dir)
    if truth is not None:
        mask = truth.notna().to_numpy()
        test_metrics = full_metrics(truth[mask].astype(int).to_numpy(), probs[mask])
        test_metrics["scored_rows"] = int(mask.sum())
        logger.info("测试集（本地公开标签 %d 条）：%s", int(mask.sum()), test_metrics)

    # ---- 落盘 ----
    ckpt = Path(args.ckpt_dir) / tag
    ckpt.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(ckpt))
    bundle.tokenizer.save_pretrained(str(ckpt))
    model_lib.save_moco_state(bundle, str(ckpt / "moco_state.pt"))

    if trainer.step_log:
        path = out_dir / f"{tag}_steps.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trainer.step_log[0]))
            writer.writeheader()
            writer.writerows(trainer.step_log)
        logger.info("已写出 %s（%d 行对比统计）", path, len(trainer.step_log))

    history = [e for e in trainer.state.log_history if "loss" in e or "eval_loss" in e]
    (out_dir / f"{tag}_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    evals = [e for e in trainer.state.log_history if "eval_accuracy" in e]
    best = max(evals, key=lambda e: e["eval_accuracy"]) if evals else {}
    last_log = trainer.step_log[-1] if trainer.step_log else {}
    summary = {
        "tag": tag,
        "method": args.method,
        "model_id": args.model_id,
        "config": {**vars(args), "grad_accum_resolved": accum,
                   "effective_batch": args.batch_size * accum},
        "val_metrics_last_epoch": {k: round(float(v), 4) for k, v in val_metrics.items()
                                   if isinstance(v, (int, float))},
        "val_best_epoch": best.get("epoch"),
        "val_best_accuracy": best.get("eval_accuracy"),
        "val_per_epoch": [{k: e.get(k) for k in
                           ("epoch", "eval_loss", "eval_accuracy", "eval_macro_f1",
                            "eval_roc_auc")} for e in evals],
        "test_metrics": test_metrics,
        "train_seconds": round(train_seconds, 1),
        "peak_gpu_gb": round(peak_gb, 2),
        "final_ce_loss": last_log.get("ce_loss"),
        "final_contrastive_loss": last_log.get("contrastive_loss"),
        "final_total_loss": last_log.get("total_loss"),
        "mean_pos_per_query": last_log.get("pos_per_query"),
        "mean_neg_per_query": last_log.get("neg_per_query"),
        "mean_candidates_per_query": last_log.get("n_candidates"),
        # 动量分支是否在做无用功的两个诊断量，见 trainer_scl._record_progress
        "final_query_key_cosine": last_log.get("view_cos"),
        "final_contrast_progress": last_log.get("contrast_progress"),
        "queue_valid_end": int(bundle.queue.valid.item()) if bundle.queue else None,
        "queue_label_counts_end": bundle.queue.label_counts() if bundle.queue else None,
        "ema_calls": trainer.mstate.ema_calls,
        "optimizer_steps": trainer.state.global_step,
        "checkpoint": str(ckpt),
    }
    (out_dir / f"{tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写出 %s", out_dir / f"{tag}_summary.json")
    bundle.close()
    return summary


if __name__ == "__main__":
    main()
