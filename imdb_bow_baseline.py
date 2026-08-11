"""任务 12 的传统分类器基线：Bag of Words / TF-IDF + 传统模型。

对比表里需要一行「传统分类器 + Bag of Words」。这个脚本把它跑出来，并且写成
和神经网络模型完全一样的 `results/<name>_summary.json` 格式，
这样 `tools/collect_results.py` 能把两类模型放进同一张表。

**划分方式与 imdb_process.py 完全一致**：`train_test_split(test_size=0.2,
random_state=42, stratify=y)`，同样 20,000 训练 / 5,000 验证。传入的样本数、
random_state 和 stratify 都相同，sklearn 给出的索引就是同一套——
两类模型的验证准确率因此可以直接比较。

跑两个基线：

* `bow_rf`    —— CountVectorizer(5000) + 随机森林。这就是 Kaggle 教程 Part 1 的模型；
* `tfidf_lr`  —— TF-IDF(1-2gram, 20 万特征) + 逻辑回归。加这一行是为了说明
  「稀疏特征天花板低」是错觉：换成 TF-IDF + bigram + 线性模型，同样的稀疏思路
  就能追上甚至超过部分神经网络。

用法：

    python imdb_bow_baseline.py
    python imdb_bow_baseline.py --only bow_rf
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import common

BASELINES = {
    "bow_rf": {
        "label": "随机森林",
        "representation": "Bag of Words (5,000 词频)",
        "vectorizer": lambda: CountVectorizer(max_features=5_000),
        "model": lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                random_state=common.SEED),
    },
    "tfidf_lr": {
        "label": "逻辑回归",
        "representation": "TF-IDF (1-2gram, 200,000)",
        "vectorizer": lambda: TfidfVectorizer(max_features=200_000, ngram_range=(1, 2),
                                              sublinear_tf=True),
        "model": lambda: LogisticRegression(C=10.0, max_iter=2_000, n_jobs=-1),
    },
}


def run_baseline(name: str, spec: dict, train_text: list[str], labels: np.ndarray,
                 test_text: list[str], test_ids: pd.Series) -> dict:
    logging.info("=== %s：%s + %s ===", name, spec["representation"], spec["label"])

    # 词表只在训练分片上 fit：否则验证集的词频分布会泄漏进特征空间，指标偏高。
    fit_text, val_text, fit_y, val_y = train_test_split(
        train_text, labels, test_size=0.2, random_state=common.SEED, stratify=labels)

    started = time.time()
    vectorizer = spec["vectorizer"]()
    fit_x = vectorizer.fit_transform(fit_text)
    val_x = vectorizer.transform(val_text)
    model = spec["model"]()
    model.fit(fit_x, fit_y)
    elapsed = time.time() - started

    predictions = model.predict(val_x)
    probabilities = model.predict_proba(val_x)[:, 1]
    accuracy = accuracy_score(val_y, predictions)
    auc = roc_auc_score(val_y, probabilities)

    density = fit_x.nnz / (fit_x.shape[0] * fit_x.shape[1])
    logging.info("特征矩阵 %s，稀疏度 %.4f%%", fit_x.shape, 100 * density)
    logging.info("val_acc %.4f  roc_auc %.4f  训练 %.1fs", accuracy, auc, elapsed)

    # 提交文件用全部 25,000 条标注数据重训——留出集只用于选型，不该浪费数据。
    full_vectorizer = spec["vectorizer"]()
    full_x = full_vectorizer.fit_transform(train_text)
    test_x = full_vectorizer.transform(test_text)
    full_model = spec["model"]()
    full_model.fit(full_x, labels)
    # 交正面概率而不是 0/1：竞赛指标是 ROC-AUC，硬标签把概率信息全丢了。
    # 标题行按官方 sampleSubmission.csv 带引号写。
    test_probabilities = full_model.predict_proba(test_x)[:, 1]
    submission_path = common.RESULTS_DIR / f"{name}_submission.csv"
    with submission_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('"id","sentiment"\n')
        for identifier, probability in zip(test_ids, test_probabilities):
            handle.write(f"{identifier},{probability:.6f}\n")
    logging.info("已写出 %s（%d 行，正面概率）", submission_path.name, len(test_ids))

    summary = {
        "model": name,
        "text_representation": spec["representation"],
        "classifier": spec["label"],
        "best_val_acc": round(float(accuracy), 4),
        "roc_auc": round(float(auc), 4),
        "best_epoch": None,
        "epochs": None,
        "total_params": int(fit_x.shape[1]),          # 稀疏模型的「参数」就是特征维度
        "trainable_params": int(fit_x.shape[1]),
        "train_seconds": round(elapsed, 1),
        "seconds_per_epoch": None,
        "device": "cpu",
        "checkpoint": None,
    }
    path = common.RESULTS_DIR / f"{name}_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("已写出 %s", path.name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(BASELINES), help="只跑其中一个基线")
    args = parser.parse_args()

    common.setup("bow_baseline")
    train = pd.read_csv(common.CORPUS_DIR / "labeledTrainData.tsv", header=0,
                        delimiter="\t", quoting=csv.QUOTE_NONE)
    test = pd.read_csv(common.CORPUS_DIR / "testData.tsv", header=0,
                       delimiter="\t", quoting=csv.QUOTE_NONE)

    # 清洗与分词沿用 common.review_to_wordlist，和所有神经网络模型完全一致，
    # 保证差异只来自「特征表示 + 模型」，而不是预处理。
    train_text = [" ".join(common.review_to_wordlist(r)) for r in train["review"]]
    test_text = [" ".join(common.review_to_wordlist(r)) for r in test["review"]]
    labels = train["sentiment"].to_numpy()

    names = [args.only] if args.only else list(BASELINES)
    for name in names:
        run_baseline(name, BASELINES[name], train_text, labels, test_text, test["id"])


if __name__ == "__main__":
    main()
