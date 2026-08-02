"""Part 1：Bag of Words + Random Forest。

对应原教程 Part 1。流程：清洗影评 → CountVectorizer 转词频向量 → 随机森林分类
→ 生成 Kaggle 提交文件。

    python src/part1_bag_of_words.py --output submissions/part1_bow.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from imdb import clean_review, find_data_dir, read_tsv

RANDOM_STATE = 42
MAX_FEATURES = 5_000
N_ESTIMATORS = 100


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def evaluate(clean_train: pd.Series, labels: pd.Series) -> dict[str, float]:
    """在留出集上评估。

    词表只在训练分片上 `fit`，验证分片只 `transform`——否则验证集的词频信息会
    泄漏进特征空间，指标会偏高。
    """
    train_text, valid_text, train_y, valid_y = train_test_split(
        clean_train, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    vectorizer = CountVectorizer(max_features=MAX_FEATURES)
    train_x = vectorizer.fit_transform(train_text)
    valid_x = vectorizer.transform(valid_text)

    model = build_model()
    model.fit(train_x, train_y)
    predictions = model.predict(valid_x)
    probabilities = model.predict_proba(valid_x)[:, 1]

    print(f"词表大小: {len(vectorizer.get_feature_names_out()):,}")
    print(f"特征矩阵: {train_x.shape}，稀疏度 {train_x.nnz / (train_x.shape[0] * train_x.shape[1]):.4%}")
    print(classification_report(valid_y, predictions, digits=4, zero_division=0))
    return {
        "accuracy": accuracy_score(valid_y, predictions),
        "roc_auc": roc_auc_score(valid_y, probabilities),
    }


def predict_test(
    clean_train: pd.Series, labels: pd.Series, clean_test: pd.Series, test_ids: pd.Series
) -> pd.DataFrame:
    """用全部标注数据重训，再预测测试集。"""
    vectorizer = CountVectorizer(max_features=MAX_FEATURES)
    train_x = vectorizer.fit_transform(clean_train)
    test_x = vectorizer.transform(clean_test)

    model = build_model()
    model.fit(train_x, labels)
    return pd.DataFrame({"id": test_ids, "sentiment": model.predict(test_x)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("submissions/part1_bow.csv"))
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    train = read_tsv(data_dir, "labeledTrainData")
    test = read_tsv(data_dir, "testData")
    print(f"数据目录: {data_dir}  训练 {len(train):,} 条，测试 {len(test):,} 条")

    print("\n[1/3] 清洗文本")
    clean_train = train["review"].map(clean_review)
    clean_test = test["review"].map(clean_review)

    print("\n[2/3] 留出集评估")
    metrics = evaluate(clean_train, train["sentiment"])
    print(f"Accuracy: {metrics['accuracy']:.4f}   ROC-AUC: {metrics['roc_auc']:.4f}")

    print("\n[3/3] 全量重训并生成提交文件")
    submission = predict_test(clean_train, train["sentiment"], clean_test, test["id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, quoting=csv.QUOTE_NONE)
    print(f"已写出 {args.output}")


if __name__ == "__main__":
    main()
