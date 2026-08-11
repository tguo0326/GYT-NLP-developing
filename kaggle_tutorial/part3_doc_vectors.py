"""Part 3：把词向量聚合成文档向量，再做分类。

对应原教程 Part 3。Word2Vec 给的是「词」的向量，分类器需要的是「整条影评」的
向量。本文件实现原教程的两种聚合方式：

  average  —— 对影评里所有词的向量取平均，得到 300 维稠密向量。
  cluster  —— 先用 K-Means 把词表聚成若干「语义簇」（原教程称 bag of centroids），
              再统计每条影评落在各簇的词数，得到稀疏的簇计数向量。

对比 Part 1 的 BoW，可以看清一件事：BoW 的每一维是「某个具体词出现几次」，
average 向量的每一维没有可读的词义，是分布式表示里的一个坐标分量。这正是从
稀疏离散表示走向稠密 Embedding 的分界点。

    python kaggle_tutorial/part3_doc_vectors.py --method average
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from imdb import find_data_dir, read_tsv, review_to_words

RANDOM_STATE = 42
N_ESTIMATORS = 100
# 原教程取「词表大小 / 5」，即平均每簇约 5 个词。
WORDS_PER_CLUSTER = 5


def average_vectors(reviews: pd.Series, model: Word2Vec) -> np.ndarray:
    """每条影评 = 其词向量的算术平均。

    不在词表里的词直接跳过（min_count=40 过滤掉了低频词）。整条影评一个词都不
    命中时返回零向量——这种情况在 25,000 条里极少，但必须显式处理，否则会除零。
    """
    vectors = model.wv
    features = np.zeros((len(reviews), vectors.vector_size), dtype=np.float32)

    for row, review in enumerate(reviews):
        known = [word for word in review_to_words(review, remove_stopwords=True) if word in vectors]
        if known:
            features[row] = np.mean(vectors[known], axis=0)
    return features


def build_word_clusters(model: Word2Vec) -> dict[str, int]:
    """对词向量做 K-Means，返回 词 → 簇编号 的映射。"""
    vectors = model.wv
    n_clusters = max(2, len(vectors.index_to_key) // WORDS_PER_CLUSTER)
    print(f"K-Means: {len(vectors.index_to_key):,} 个词 → {n_clusters:,} 个簇")

    # 原教程用 KMeans，词表上万时很慢；MiniBatchKMeans 结果接近但快一个量级。
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=3, batch_size=1024
    )
    assignments = kmeans.fit_predict(vectors.vectors)
    return dict(zip(vectors.index_to_key, assignments.tolist()))


def centroid_vectors(reviews: pd.Series, word_to_cluster: dict[str, int]) -> np.ndarray:
    """每条影评 = 各语义簇的词计数（bag of centroids）。"""
    n_clusters = max(word_to_cluster.values()) + 1
    features = np.zeros((len(reviews), n_clusters), dtype=np.float32)

    for row, review in enumerate(reviews):
        for word in review_to_words(review, remove_stopwords=True):
            cluster = word_to_cluster.get(word)
            if cluster is not None:
                features[row, cluster] += 1
    return features


def evaluate(features: np.ndarray, labels: pd.Series) -> dict[str, float]:
    train_x, valid_x, train_y, valid_y = train_test_split(
        features, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=RANDOM_STATE
    )
    model.fit(train_x, train_y)
    predictions = model.predict(valid_x)
    probabilities = model.predict_proba(valid_x)[:, 1]

    print(classification_report(valid_y, predictions, digits=4, zero_division=0))
    return {
        "accuracy": accuracy_score(valid_y, predictions),
        "roc_auc": roc_auc_score(valid_y, probabilities),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=Path("models/word2vec_300d.model"))
    parser.add_argument("--method", choices=("average", "cluster"), default="average")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"缺少词向量模型 {args.model}，请先运行 kaggle_tutorial/part2_word2vec.py")

    data_dir = find_data_dir(args.data_dir)
    train = read_tsv(data_dir, "labeledTrainData")
    test = read_tsv(data_dir, "testData")
    word2vec = Word2Vec.load(str(args.model))
    print(f"载入词向量：{len(word2vec.wv):,} 词 × {word2vec.wv.vector_size} 维")

    print(f"\n[1/3] 构造文档向量（method={args.method}）")
    if args.method == "average":
        train_features = average_vectors(train["review"], word2vec)
        test_features = average_vectors(test["review"], word2vec)
    else:
        word_to_cluster = build_word_clusters(word2vec)
        train_features = centroid_vectors(train["review"], word_to_cluster)
        test_features = centroid_vectors(test["review"], word_to_cluster)
    print(f"训练特征矩阵：{train_features.shape}")

    print("\n[2/3] 留出集评估")
    metrics = evaluate(train_features, train["sentiment"])
    print(f"Accuracy: {metrics['accuracy']:.4f}   ROC-AUC: {metrics['roc_auc']:.4f}")

    print("\n[3/3] 全量重训并生成提交文件")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=RANDOM_STATE
    )
    model.fit(train_features, train["sentiment"])
    submission = pd.DataFrame(
        {"id": test["id"], "sentiment": model.predict(test_features)}
    )
    output = args.output or Path(f"submissions/part3_{args.method}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False, quoting=csv.QUOTE_NONE)
    print(f"已写出 {output}")


if __name__ == "__main__":
    main()
