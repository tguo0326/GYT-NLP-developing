"""所有模型脚本共用的训练基础设施（任务 6）。

原始代码里每个 `imdb_*.py` 都自己抄一遍训练循环，同样的问题也就重复了六遍：
写死 `device = torch.device('cuda:0')` 和 `.cuda()`（没有 GPU 直接崩）、
不切 `model.train()` / `model.eval()`（dropout 在验证时仍然生效）、
`train_loss += loss` 累加的是带计算图的张量（显存持续增长）、
不固定随机种子、不保存最佳模型、跑完什么都不留下。

这里把这些统一收进一个模块，模型脚本只负责定义网络结构和超参数。

对外接口：

    load_data()                 读取 pickle/imdb_glove.pickle3
    setup(name)                 固定种子 + 选设备 + 配置日志（返回 device）
    count_parameters(model)     参数量统计
    train(...)                  完整训练/验证循环，保存最佳权重与日志
    predict_reviews(...)        对任意新评论文本做预测
    write_submission(...)       生成 Kaggle 提交文件
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SEED = 42
MAX_LEN = 512
EMBED_SIZE = 300

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus" / "imdb"
PICKLE_PATH = ROOT / "pickle" / "imdb_glove.pickle3"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

# 文本清洗：原代码用 BeautifulSoup(review, "lxml").get_text() 去 HTML。
# IMDB 影评里的「HTML」只有 <br /> 和少量实体，正则等价且快一个数量级
# （10 万条从约 3 分钟降到约 10 秒），也不会触发 bs4 的 MarkupResemblesLocator 警告。
HTML_TAG = re.compile(r"<[^>]+>")
NON_LETTER = re.compile(r"[^a-zA-Z]")


def review_to_wordlist(review: str) -> list[str]:
    """去 HTML → 去非字母 → 小写 → 空格切分。与原 `review_to_wordlist` 行为一致。"""
    text = HTML_TAG.sub(" ", unescape(str(review)))
    return NON_LETTER.sub(" ", text).lower().split()


@dataclass
class Bundle:
    """imdb_process.py 产出的全部数据。字段顺序与 pickle 里的列表一一对应。"""

    train_features: torch.Tensor
    train_labels: torch.Tensor
    val_features: torch.Tensor
    val_labels: torch.Tensor
    test_features: torch.Tensor
    weight: torch.Tensor
    word_to_idx: dict[str, int]
    idx_to_word: dict[int, str]
    vocab: set[str]


def load_data(path: Path = PICKLE_PATH) -> Bundle:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。请先运行：python imdb_process.py"
        )
    logging.info("loading %s ...", path)
    with path.open("rb") as handle:
        return Bundle(*pickle.load(handle))


def set_seed(seed: int = SEED) -> None:
    """固定随机种子。cudnn.deterministic 会牺牲一点速度换取可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """自动选择 CPU / GPU，替代写死的 torch.device('cuda:0')。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup(name: str, seed: int = SEED) -> torch.device:
    """每个脚本开头调用一次：日志、种子、设备。"""
    LOGS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    logging.root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / f"{name}.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    set_seed(seed)
    device = get_device()
    logging.info("model=%s device=%s seed=%d", name, device, seed)
    if device.type == "cuda":
        logging.info("gpu: %s", torch.cuda.get_device_name(0))
    return device


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。Embedding 被冻结，两者差就是词向量矩阵。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
    clip: float | None,
) -> tuple[float, float]:
    """跑一个 epoch。optimizer 为 None 时是验证模式。返回 (平均损失, 准确率)。"""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, seen = 0.0, 0, 0
    # 非交互式运行（重定向到文件、nohup）时关掉进度条，否则日志里会塞满几十万行
    # 回车刷新的残片。真正的指标每个 epoch 由 logging 记录一次。
    bar = tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty())
    # 验证阶段整段包在 no_grad 里：不建计算图，显存占用和耗时都减半。
    with torch.set_grad_enabled(training):
        for features, labels in bar:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
            scores = model(features)
            loss = criterion(scores, labels)
            if training:
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            batch = labels.size(0)
            # .item() 取标量：原代码 `train_loss += loss` 累加的是张量，
            # 整个 epoch 的计算图都被挂住不释放。
            total_loss += loss.item() * batch
            correct += (scores.argmax(dim=1) == labels).sum().item()
            seen += batch
            bar.set_postfix(loss=f"{total_loss / seen:.4f}", acc=f"{correct / seen:.4f}")

    return total_loss / seen, correct / seen


def train(
    name: str,
    model: nn.Module,
    bundle: Bundle,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    *,
    num_epochs: int = 10,
    batch_size: int = 64,
    criterion: nn.Module | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    clip: float | None = 5.0,
) -> dict:
    """完整训练/验证循环。

    每个 epoch 记录训练损失、验证损失和准确率；验证准确率创新高就把权重存到
    `models/<name>_best.pt`。训练结束后自动载回最佳权重，写出 history CSV
    与 summary JSON，并生成 Kaggle 提交文件。
    """
    criterion = criterion or nn.CrossEntropyLoss()
    model.to(device)

    total_params, trainable_params = count_parameters(model)
    logging.info("参数量: 总计 %s, 可训练 %s", f"{total_params:,}", f"{trainable_params:,}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(bundle.train_features, bundle.train_labels),
        batch_size=batch_size, shuffle=True, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(bundle.val_features, bundle.val_labels),
        batch_size=batch_size, shuffle=False, pin_memory=pin,
    )

    ckpt_path = MODELS_DIR / f"{name}_best.pt"
    history, best_acc, best_epoch, started = [], -1.0, -1, time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer, f"{name} epoch {epoch}", clip
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, device, None, f"{name} val {epoch}", None
        )
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - epoch_start

        logging.info(
            "epoch %2d/%d  train_loss %.4f  train_acc %.4f  val_loss %.4f  val_acc %.4f  %.1fs",
            epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, elapsed,
        )
        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 6), "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 6), "seconds": round(elapsed, 2),
        })

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), ckpt_path)
            logging.info("  ↑ 验证准确率新高，已保存 %s", ckpt_path)

    total_time = time.time() - started
    logging.info("训练结束：最佳 val_acc %.4f（epoch %d），总耗时 %.1fs",
                 best_acc, best_epoch, total_time)

    # 载回最佳权重——最后一个 epoch 通常已经过拟合，后续预测都用最佳那份。
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    history_path = RESULTS_DIR / f"{name}_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "model": name,
        "text_representation": "GloVe 840B.300d",
        "best_val_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_seconds": round(total_time, 1),
        "seconds_per_epoch": round(total_time / num_epochs, 1),
        "device": str(device),
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
    }
    summary_path = RESULTS_DIR / f"{name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("已写出 %s 与 %s", history_path.name, summary_path.name)
    return summary


@torch.no_grad()
def predict_logits(model: nn.Module, features: torch.Tensor, device: torch.device,
                   batch_size: int = 128) -> torch.Tensor:
    """批量前向，返回 CPU 上的 logits。"""
    model.eval()
    outputs = []
    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size].to(device)
        outputs.append(model(batch).cpu())
    return torch.cat(outputs)


def write_submission(name: str, model: nn.Module, bundle: Bundle, device: torch.device) -> Path:
    """用最佳模型对 testData.tsv 打分，生成 Kaggle 提交文件。

    两个容易踩的坑：

    1. **不能复用 `bundle.test_features`**。它是 `imdb_process.py` 跑的时候按当时那份
       testData.tsv 的行顺序编码的。Kaggle 官方文件和从 aclImdb 重建的版本行数都是
       25,000 但顺序完全不同——复用就会拿 A 的预测配 B 的 id，文件格式看着正常、
       分数等于随机。所以一律按当前 TSV 重新编码。
    2. **交概率而不是 0/1**。竞赛指标是 ROC-AUC，硬标签把概率信息全丢了，
       实测能差好几个百分点。
    """
    test = pd.read_csv(CORPUS_DIR / "testData.tsv", header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
    features = encode_texts(test["review"].tolist(), bundle.word_to_idx)
    probabilities = torch.softmax(predict_logits(model, features, device), dim=1)[:, 1].numpy()

    path = RESULTS_DIR / f"{name}_submission.csv"
    # 标题行按官方 sampleSubmission.csv 的写法带引号
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('"id","sentiment"\n')
        for identifier, probability in zip(test["id"], probabilities):
            handle.write(f"{identifier},{probability:.6f}\n")
    logging.info("已写出 %s（%d 行，正面概率）", path.name, len(test))
    return path


def encode_texts(texts: list[str], word_to_idx: dict[str, int], maxlen: int = MAX_LEN) -> torch.Tensor:
    """把原始评论文本编码成定长 id 序列，与训练时的处理完全一致。"""
    rows = []
    for text in texts:
        ids = [word_to_idx.get(token, 0) for token in review_to_wordlist(text)][:maxlen]
        rows.append(ids + [0] * (maxlen - len(ids)))
    return torch.tensor(rows, dtype=torch.long)


def predict_reviews(model: nn.Module, texts: list[str], bundle: Bundle,
                    device: torch.device) -> list[dict]:
    """任务 6 的「支持输入一条新评论进行预测」。返回标签与正面概率。"""
    features = encode_texts(texts, bundle.word_to_idx)
    probs = torch.softmax(predict_logits(model, features, device), dim=1)
    results = []
    for text, prob in zip(texts, probs):
        label = int(prob.argmax())
        results.append({
            "review": text,
            "label": label,
            "sentiment": "positive" if label == 1 else "negative",
            "prob_positive": round(float(prob[1]), 4),
        })
    return results


# 交互式/命令行预测时的默认样例，也用于训练结束后的冒烟测试。
DEMO_REVIEWS = [
    "This movie was absolutely wonderful. The acting was superb and the story moved me to tears.",
    "A complete waste of time. Terrible script, wooden acting, and the plot made no sense at all.",
]


def report_demo_predictions(model: nn.Module, bundle: Bundle, device: torch.device,
                            texts: list[str] | None = None) -> None:
    for item in predict_reviews(model, texts or DEMO_REVIEWS, bundle, device):
        logging.info("[%s p=%.4f] %s", item["sentiment"], item["prob_positive"],
                     item["review"][:80])


def build_parser(*, epochs: int = 10, batch_size: int = 64, lr: float = 1e-3,
                 description: str | None = None) -> argparse.ArgumentParser:
    """所有模型脚本共用的命令行参数。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--clip", type=float, default=5.0, help="梯度裁剪阈值，0 表示不裁剪")
    parser.add_argument("--predict", nargs="+", metavar="REVIEW",
                        help="跳过训练：载入已保存的最佳模型，对给定评论做预测")
    parser.add_argument("--no-submission", action="store_true", help="不生成 Kaggle 提交文件")
    return parser


def load_best(name: str, model: nn.Module, device: torch.device) -> nn.Module:
    path = MODELS_DIR / f"{name}_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}。请先训练：python imdb_{name}.py")
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


@dataclass
class RunResult:
    """run() 的返回值。Attention 等需要在训练后继续做分析的脚本会用到。"""

    model: nn.Module
    bundle: Bundle
    device: torch.device
    summary: dict | None


def run(name: str, args: argparse.Namespace, make_model, make_optimizer) -> RunResult:
    """标准流程：准备 → 建模 →（预测 | 训练）→ 提交文件 → 新评论冒烟测试。

    `make_model(bundle)` 返回未 to(device) 的模型；`make_optimizer(model)` 返回优化器。
    带 `--predict` 时只做推理，不训练。
    """
    device = setup(name, args.seed)
    bundle = load_data()
    model = make_model(bundle).to(device)
    total, trainable = count_parameters(model)
    logging.info("%s: 总参数 %s / 可训练 %s", name, f"{total:,}", f"{trainable:,}")

    if args.predict:
        load_best(name, model, device)
        for item in predict_reviews(model, list(args.predict), bundle, device):
            logging.info("[%s p=%.4f] %s", item["sentiment"], item["prob_positive"], item["review"])
        return RunResult(model, bundle, device, None)

    summary = train(
        name, model, bundle, device, make_optimizer(model),
        num_epochs=args.epochs, batch_size=args.batch_size,
        clip=args.clip if args.clip and args.clip > 0 else None,
    )
    if not args.no_submission:
        write_submission(name, model, bundle, device)
    report_demo_predictions(model, bundle, device)
    return RunResult(model, bundle, device, summary)
