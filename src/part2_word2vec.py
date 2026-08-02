"""Part 2：训练 Word2Vec 词向量。

对应原教程 Part 2。用 75,000 条影评（25,000 标注 + 50,000 无标注）训练 gensim
Word2Vec，得到 300 维稠密词向量并保存到磁盘，供 Part 3 复用。

Word2Vec 的训练目标不需要人工标签：它用「上下文预测中心词」（CBOW）或
「中心词预测上下文」（Skip-gram）构造监督信号，标签直接来自语料本身。这就是
自监督学习，也是后来 BERT / GPT 系列预训练的同一条思路。

    python src/part2_word2vec.py --output models/word2vec_300d.model
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gensim.models import Word2Vec

from imdb import find_data_dir, read_tsv, review_to_sentences

# 原教程的超参数，保持一致以便对照。
VECTOR_SIZE = 300      # 词向量维度
MIN_COUNT = 40         # 词频下限，过滤拼写错误和极低频词
WINDOW = 10            # 上下文窗口半径
DOWNSAMPLE = 1e-3      # 高频词降采样阈值
EPOCHS = 5
SEED = 42


def collect_sentences(data_dir: Path, use_unlabeled: bool = True) -> list[list[str]]:
    """把影评切成句子列表。

    这里刻意把标注和无标注数据一起用上：Word2Vec 训练不看 sentiment 标签，
    所以 50,000 条无标注影评是「免费」的额外语料。
    """
    sentences: list[list[str]] = []

    labeled = read_tsv(data_dir, "labeledTrainData")
    print(f"标注影评 {len(labeled):,} 条 → 分句")
    for review in labeled["review"]:
        sentences.extend(review_to_sentences(review))

    if use_unlabeled:
        unlabeled = read_tsv(data_dir, "unlabeledTrainData")
        print(f"无标注影评 {len(unlabeled):,} 条 → 分句")
        for review in unlabeled["review"]:
            sentences.extend(review_to_sentences(review))

    print(f"共 {len(sentences):,} 句，{sum(len(s) for s in sentences):,} 词")
    return sentences


def train(sentences: list[list[str]], workers: int, sg: int) -> Word2Vec:
    """训练 Word2Vec。

    `workers > 1` 时 gensim 的多线程调度不保证确定性，即使固定 seed 也可能有
    微小差异；需要严格复现时把 workers 设为 1。
    """
    return Word2Vec(
        sentences,
        vector_size=VECTOR_SIZE,
        min_count=MIN_COUNT,
        window=WINDOW,
        sample=DOWNSAMPLE,
        sg=sg,
        epochs=EPOCHS,
        workers=workers,
        seed=SEED,
    )


def inspect(model: Word2Vec) -> None:
    """定性检查词向量学到了什么。"""
    vectors = model.wv
    print(f"\n词表 {len(vectors):,} 个词，每个词 {vectors.vector_size} 维")

    print("\n最近邻：")
    for word in ["awful", "brilliant", "actress", "france"]:
        if word in vectors:
            neighbours = ", ".join(w for w, _ in vectors.most_similar(word, topn=6))
            print(f"  {word:10s} → {neighbours}")

    print("\n找出不同类的词：")
    for group in (["kitchen", "bedroom", "bathroom", "france"], ["awful", "terrible", "dreadful", "great"]):
        if all(word in vectors for word in group):
            print(f"  {group} → {vectors.doesnt_match(group)}")

    print("\n向量算术（king - man + woman）：")
    if all(word in vectors for word in ("king", "man", "woman")):
        result = vectors.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
        print("  " + ", ".join(f"{w} ({score:.3f})" for w, score in result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("models/word2vec_300d.model"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-unlabeled", action="store_true", help="只用标注数据训练（更快）")
    parser.add_argument("--sg", type=int, choices=(0, 1), default=0, help="0=CBOW，1=Skip-gram")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    data_dir = find_data_dir(args.data_dir)
    sentences = collect_sentences(data_dir, use_unlabeled=not args.skip_unlabeled)

    print(f"\n训练 Word2Vec（{'Skip-gram' if args.sg else 'CBOW'}，{VECTOR_SIZE} 维）")
    model = train(sentences, workers=args.workers, sg=args.sg)
    inspect(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"\n已保存 {args.output}")


if __name__ == "__main__":
    main()
