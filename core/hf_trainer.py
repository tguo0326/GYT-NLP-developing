"""任务 11（选做）：BERT / DistilBERT / RoBERTa 微调的共用实现。

三个脚本 `experiments/finetune/bert.py` / `experiments/finetune/distilbert.py` /
`experiments/finetune/roberta.py` 只有「模型名 + 批大小 + 学习率」不同，
所以实现收在这里，各脚本只填一份配置。

**和前面 GloVe 模型的本质区别**：GloVe 是「静态词向量」——`bank` 在
`river bank` 和 `bank account` 里是同一个向量。BERT 是「上下文词向量」——
同一个词在不同句子里向量不同，而且整个 12 层编码器都参与微调，
不是只训练顶上一个分类头。代价是参数量从百万级跳到亿级。

相比原始版本（docs/original_code/imdb_*_trainer.py）修的兼容性与正确性问题：

1. `datasets.load_metric()` 在 datasets 2.x 起就被移除了（迁到独立的 `evaluate` 包）。
   这里直接用 numpy 算准确率，少一个依赖；
2. `TrainingArguments(evaluation_strategy=...)` 在 transformers 4.46 起改名
   `eval_strategy`，旧名直接 TypeError；
3. `Trainer(tokenizer=...)` 已废弃，改成 `processing_class=`；
4. `train_test_split(train, test_size=.2)` 没有 `random_state`，每次划分都不同，
   而且和 experiments/preprocess.py 的划分不一致——三份 BERT 结果彼此、以及和 CNN/LSTM
   都不可比。这里统一成 `random_state=42, stratify=sentiment`；
5. `output_dir='./results'`：HF Trainer 会往这个目录写 checkpoint，
   会污染我们存对比结果的 `results/`。改到 `models/<name>_hf/`；
6. 结果写 `./result/`（目录不存在，跑到最后一步才崩）→ 统一 `results/`；
7. 补上 `results/<name>_summary.json`，让 `tools/collect_results.py` 能把
   BERT 系列和 GloVe 系列放进同一张对比表。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# 必须在 import transformers 之前设置。transformers 会去探测 TensorFlow，
# 而如果环境里装的是 Keras 3（TF 2.16+ 的默认），探测会直接抛
# 「Your currently installed version of Keras is Keras 3, but this is not yet
# supported in Transformers」。本项目只用 PyTorch 后端，直接关掉 TF/Flax 探测。
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from core import common


def build_parser(*, model_id: str, batch_size: int, lr: float,
                 epochs: int = 2) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=model_id)
    parser.add_argument("--epochs", type=float, default=epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--max-length", type=int, default=256,
                        help="截断长度。IMDB 评论中位数 177 词，256 个 subword 覆盖大多数")
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available(),
                        help="GPU 上默认开混合精度，能省一半显存")
    parser.add_argument("--no-submission", action="store_true")
    parser.add_argument("--predict", nargs="+", metavar="REVIEW")
    return parser


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(common.CORPUS_DIR / "labeledTrainData.tsv", header=0,
                        delimiter="\t", quoting=csv.QUOTE_NONE)
    test = pd.read_csv(common.CORPUS_DIR / "testData.tsv", header=0,
                       delimiter="\t", quoting=csv.QUOTE_NONE)
    return train, test


def run(name: str, args: argparse.Namespace) -> dict | None:
    # transformers 是可选依赖，只有真正跑 BERT 时才 import
    import datasets
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    device = common.setup(name, args.seed)
    logging.info("微调 %s（max_length=%d, batch=%d, lr=%g, epochs=%s）",
                 args.model_id, args.max_length, args.batch_size, args.lr, args.epochs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    checkpoint_dir = common.MODELS_DIR / f"{name}_hf"

    if args.predict:
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir)).to(device)
        model.eval()
        batch = tokenizer(list(args.predict), truncation=True, max_length=args.max_length,
                          padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**batch).logits, dim=1).cpu()
        for text, prob in zip(args.predict, probs):
            label = "positive" if int(prob.argmax()) == 1 else "negative"
            logging.info("[%s p=%.4f] %s", label, float(prob[1]), text)
        return None

    train_frame, test_frame = _load_frames()
    # 与 experiments/preprocess.py 完全一致的划分，保证跨模型可比
    fit_frame, val_frame = train_test_split(
        train_frame, test_size=0.2, random_state=args.seed,
        stratify=train_frame["sentiment"])
    logging.info("划分：训练 %d 条 / 验证 %d 条", len(fit_frame), len(val_frame))

    def to_dataset(frame: pd.DataFrame, labelled: bool = True) -> "datasets.Dataset":
        payload = {"text": frame["review"].tolist()}
        if labelled:
            payload["label"] = frame["sentiment"].tolist()
        return datasets.Dataset.from_dict(payload)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    tokenized_train = to_dataset(fit_frame).map(tokenize, batched=True)
    tokenized_val = to_dataset(val_frame).map(tokenize, batched=True)
    tokenized_test = to_dataset(test_frame, labelled=False).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=2)
    total_params, trainable_params = common.count_parameters(model)
    logging.info("参数量: 总计 %s, 可训练 %s", f"{total_params:,}", f"{trainable_params:,}")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}

    training_args = TrainingArguments(
        output_dir=str(common.MODELS_DIR / f"{name}_hf_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_dir=str(common.LOGS_DIR / f"{name}_hf"),
        logging_steps=100,
        eval_strategy="epoch",          # 4.46 起 evaluation_strategy 改名
        save_strategy="no",
        seed=args.seed,
        fp16=args.fp16,
        report_to=[],
        # 和 common.py 里同样的理由：重定向到文件时 tqdm 的每次刷新都变成一行，
        # 一次微调能刷出几 MB 的进度条残片。
        disable_tqdm=not sys.stderr.isatty(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,     # 旧版是 tokenizer=，已废弃
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    started = time.time()
    trainer.train()
    elapsed = time.time() - started

    # 准确率直接取最后一条 eval 记录。不再额外调用 trainer.evaluate()——
    # 那会在 log_history 里追加一条和最后一个 epoch 完全重复的记录，
    # 日志和 history CSV 里都会出现重复行。
    evals = [entry for entry in trainer.state.log_history if "eval_accuracy" in entry]
    accuracy = float(evals[-1]["eval_accuracy"]) if evals else float(
        trainer.evaluate()["eval_accuracy"])
    logging.info("验证准确率 %.4f，训练 %.1fs", accuracy, elapsed)

    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    logging.info("已保存微调后的模型到 %s", checkpoint_dir)
    # Trainer 的中间 checkpoint 目录很大（每份都是完整权重），跑完就清掉
    shutil.rmtree(common.MODELS_DIR / f"{name}_hf_ckpt", ignore_errors=True)

    # HF Trainer 把训练损失记在自己的 state.log_history 里（每 logging_steps 一条），
    # 不经过我们的 logger。手动搬过来，否则 logs/<name>.log 里只有开头和结尾，
    # 看不到训练过程——任务 6 要求的「记录训练损失、验证损失和准确率」就落空了。
    history = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            logging.info("step %-6s epoch %.2f  train_loss %.4f  lr %.2e",
                         entry.get("step"), entry.get("epoch", 0.0), entry["loss"],
                         entry.get("learning_rate", 0.0))
        if "eval_accuracy" in entry:
            logging.info("epoch %.2f  val_loss %.4f  val_acc %.4f  %.1fs",
                         entry.get("epoch", 0.0), entry.get("eval_loss", float("nan")),
                         entry["eval_accuracy"], entry.get("eval_runtime", 0.0))
            history.append({
                "epoch": round(entry.get("epoch", 0.0), 4),
                "eval_loss": entry.get("eval_loss"),
                "eval_acc": entry["eval_accuracy"],
                "eval_seconds": entry.get("eval_runtime"),
            })
    history_path = common.RESULTS_DIR / f"{name}_history.csv"
    if history:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    if not args.no_submission:
        logits = trainer.predict(tokenized_test).predictions
        submission = pd.DataFrame({"id": test_frame["id"],
                                   "sentiment": np.argmax(logits, axis=-1)})
        path = common.RESULTS_DIR / f"{name}_submission.csv"
        submission.to_csv(path, index=False, quoting=csv.QUOTE_NONE)
        logging.info("已写出 %s（%d 行）", path.name, len(submission))

    summary = {
        "model": name,
        "text_representation": f"{args.model_id} 上下文词向量（全模型微调）",
        "best_val_acc": round(accuracy, 4),
        "best_epoch": len(history) or None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_seconds": round(elapsed, 1),
        "seconds_per_epoch": round(elapsed / max(args.epochs, 1), 1),
        "device": str(device),
        "checkpoint": str(checkpoint_dir.relative_to(common.ROOT)),
    }
    summary_path = common.RESULTS_DIR / f"{name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("已写出 %s", summary_path.name)
    return summary
