"""Part 4（原教程之外的补充）：TF-IDF + 逻辑回归。

加这一节是为了给前三部分一个诚实的参照。原教程的三种方法都用随机森林，容易
让人以为「BoW 只能到 84%」。事实上把 BoW 换成 TF-IDF、加上二元词组、再换成
线性模型，同样是稀疏词频特征，准确率能明显更高——2015 年前后这类线性基线在
情感分类上长期难以被超越。

结论不是「Word2Vec 没用」，而是：
  · 在单一领域、数据量有限的分类任务上，稀疏特征 + 线性模型是很强的基线；
  · 稠密 Embedding 的优势在于迁移能力——同一份词向量可以喂给别的任务，
    而某个数据集上的 TF-IDF 词表不能。这正是预训练范式的起点。

    python kaggle_tutorial/part4_tfidf_baseline.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from imdb import clean_review, find_data_dir, read_tsv

RANDOM_STATE = 42


def build_pipeline():
    """TF-IDF（含 bigram）+ 逻辑回归。

    ngram_range=(1, 2) 让 "not good" 这类否定短语成为独立特征——纯 unigram 的
    BoW 看到 "not" 和 "good" 两个正交维度，丢掉了否定关系。
    """
    return make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
        ),
        LogisticRegression(C=10.0, max_iter=2000, random_state=RANDOM_STATE),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("submissions/part4_tfidf.csv"))
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    train = read_tsv(data_dir, "labeledTrainData")
    test = read_tsv(data_dir, "testData")

    print("[1/3] 清洗文本")
    clean_train = train["review"].map(clean_review)
    clean_test = test["review"].map(clean_review)

    print("\n[2/3] 留出集评估")
    train_text, valid_text, train_y, valid_y = train_test_split(
        clean_train,
        train["sentiment"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train["sentiment"],
    )
    pipeline = build_pipeline()
    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(valid_text)
    probabilities = pipeline.predict_proba(valid_text)[:, 1]

    n_features = len(pipeline.named_steps["tfidfvectorizer"].get_feature_names_out())
    print(f"特征数（unigram + bigram）: {n_features:,}")
    print(classification_report(valid_y, predictions, digits=4, zero_division=0))
    print(f"Accuracy: {accuracy_score(valid_y, predictions):.4f}   "
          f"ROC-AUC: {roc_auc_score(valid_y, probabilities):.4f}")

    print("\n[3/3] 全量重训并生成提交文件")
    final = build_pipeline()
    final.fit(clean_train, train["sentiment"])
    submission = pd.DataFrame(
        {"id": test["id"], "sentiment": final.predict(clean_test)}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, quoting=csv.QUOTE_NONE)
    print(f"已写出 {args.output}")


if __name__ == "__main__":
    main()
