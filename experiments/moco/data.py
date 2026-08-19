"""IMDB 数据加载 —— 逐行复刻 0.9633 那次实验的划分，保证可比。

对齐对象：`科研专用/NLP/core/peft_trainer.py::run` + `NLP/tools/score_test.py`。
四件事必须一模一样，否则六组实验之间、以及与 0.9633 之间都不可比：

1. `quoting=csv.QUOTE_NONE` 读 tsv（影评正文里有裸引号，用默认 quoting 会串行）；
2. `train_test_split(test_size=0.2, random_state=SEED, stratify=sentiment)` → 20000/5000；
3. 测试集**按文件原序**保留（`group_by_length=True` 会让 predict 重排，
   所以预测那边必须先关掉它，见 trainer_scl.predict_in_order）；
4. 本地打分先按 id 对齐，对不上退回按正文对齐（官方 Kaggle 文件把内部引号转义成
   `\"`，aclImdb 重建版是裸引号，不做归一化近半数会对不上）→ 实测 24961/25000 条。
"""
from __future__ import annotations

import csv
import logging
import os
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# 与 NLP/core/common.py 的 SEED 一致
SEED = 42
# 与仓库其余部分共用 corpus/imdb（.gitignore 里，需自己准备，见根目录 README 的「数据」）
DEFAULT_DATA_DIR = os.environ.get(
    "IMDB_DIR", str(Path(__file__).resolve().parents[2] / "corpus" / "imdb"))

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """把 Kaggle 官方文件与 aclImdb 重建版的同一条评论归一到可比形式。"""
    return _WHITESPACE.sub(" ", str(text).replace('\\"', '"')).strip().strip('"')


def load_frames(data_dir: str = DEFAULT_DATA_DIR):
    """返回 (train_frame, test_frame)。"""
    root = Path(data_dir)
    train_path, test_path = root / "labeledTrainData.tsv", root / "testData.tsv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 不存在。用 IMDB_DIR 指定 Kaggle word2vec-nlp-tutorial 的数据目录")
    train = pd.read_csv(train_path, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
    test = pd.read_csv(test_path, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
    return train, test


def load_test_labels(test_frame: pd.DataFrame, data_dir: str = DEFAULT_DATA_DIR):
    """返回与 test_frame 行对齐的真实标签 Series（含 NaN 表示对不上），没有文件则 None。"""
    path = Path(data_dir) / "testDataWithLabels.tsv"
    if not path.exists():
        logger.warning("没有 %s，无法本地打测试集分数", path)
        return None
    labelled = pd.read_csv(path, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)

    merged = test_frame[["id"]].merge(labelled[["id", "sentiment"]], on="id", how="left")
    if not merged["sentiment"].isna().any():
        return merged["sentiment"].astype(float)

    logger.info("id 与 testDataWithLabels.tsv 对不上，改按评论正文对齐")
    truth = test_frame[["review"]].assign(key=test_frame["review"].map(_normalize)).merge(
        labelled.assign(key=labelled["review"].map(_normalize))[["key", "sentiment"]]
        .drop_duplicates("key"), on="key", how="left")
    matched = int(truth["sentiment"].notna().sum())
    logger.info("正文对齐 %d / %d 条", matched, len(test_frame))
    if matched < 0.95 * len(test_frame):
        logger.warning("对齐率过低，放弃本地打分")
        return None
    return truth["sentiment"]


def build_datasets(tokenizer, data_dir: str = DEFAULT_DATA_DIR, max_length: int = 384,
                   seed: int = SEED, subset: int = 0):
    """返回 (train_ds, val_ds, test_ds, test_frame)。

    subset 只用于 smoke test；正式实验必须留 0。故意**不**截断测试集，
    这样冒烟测试也能验证「概率和 id 有没有对上」（那是原项目踩过最贵的一个坑）。
    """
    train_frame, test_frame = load_frames(data_dir)
    fit_frame, val_frame = train_test_split(
        train_frame, test_size=0.2, random_state=seed, stratify=train_frame["sentiment"])
    if subset:
        fit_frame = fit_frame.head(subset)
        val_frame = val_frame.head(max(subset // 4, 8))
        test_frame = test_frame.head(max(subset, 64))
        logger.warning("subset=%d：只是冒烟测试，结果不可汇报", subset)
    logger.info("划分：训练 %d 条 / 验证 %d 条 / 测试 %d 条",
                len(fit_frame), len(val_frame), len(test_frame))

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    def to_ds(frame, labelled=True):
        payload = {"text": frame["review"].tolist()}
        if labelled:
            payload["labels"] = frame["sentiment"].tolist()
        return Dataset.from_dict(payload).map(tokenize, batched=True,
                                              remove_columns=["text"])

    return (to_ds(fit_frame), to_ds(val_frame),
            to_ds(test_frame, labelled=False), test_frame)


def save_submission(test_frame: pd.DataFrame, probs, path: str) -> None:
    """写 Kaggle 提交文件。逐字节对齐 sampleSubmission.csv：交概率而不是硬标签
    （竞赛指标是 ROC-AUC）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('"id","sentiment"\n')
        for identifier, probability in zip(test_frame["id"], probs):
            handle.write(f"{identifier},{float(probability):.6f}\n")
    logger.info("已写出 %s（%d 行，正面概率）", out, len(probs))
