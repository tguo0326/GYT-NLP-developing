"""任务 5：IMDB 数据预处理，产出 pickle/imdb_glove.pickle3。

相比原始版本（docs/original_code/imdb_process.py）改了六处：

1. **GloVe 路径**：原代码写死 `g:\\lib\\glove.840B.300d.gensim.txt`（Windows 绝对
   路径），改成相对路径 `glove/`，并优先读 Gensim 原生 `.kv` 格式；
2. **Gensim 4.x**：原代码用 `load_word2vec_format` 读一个预先加过头部的 txt。
   这里直接支持 `tools/prepare_glove.py` 产出的 `.kv`，用 `mmap='r'` 加载，
   秒级完成而不是几分钟；
3. **Embedding 矩阵构建**：原代码遍历 GloVe 的 220 万个词去查 IMDB 词表，每命中
   一次还 `print(i)`——220 万次 I/O。改成遍历 IMDB 词表（约 10 万）去查 GloVe，
   顺带就能统计 OOV；
4. **自动建目录**：`pickle/` 不存在时原代码直接崩，现在自动创建；
5. **文本清洗**：`BeautifulSoup(review, "lxml")` 换成正则（见 common.py 说明）；
6. **可观测性**：输出数据量、词表大小、各张量形状、OOV 数量与占比。

用法：

    python experiments/preprocess.py
    python experiments/preprocess.py --max-len 512 --test-size 0.2
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许 `python experiments/.../x.py` 直接运行（不必写成 python -m）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import logging
import pickle
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

from core.common import (
    CORPUS_DIR,
    EMBED_SIZE,
    MAX_LEN,
    PICKLE_PATH,
    SEED,
    review_to_wordlist,
    set_seed,
)

GLOVE_DIR = Path("glove")
# 优先用 tools/prepare_glove.py 产出的 Gensim 原生格式；退回原始 txt。
GLOVE_CANDIDATES = ("glove.840B.300d.kv", "glove.840B.300d.txt")


def read_corpus() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读三份 TSV。quoting=QUOTE_NONE 是必须的：影评正文里全是英文引号，
    按默认规则解析会把字段吞掉。"""
    frames = []
    for stem in ("labeledTrainData", "testData", "unlabeledTrainData"):
        path = CORPUS_DIR / f"{stem}.tsv"
        if not path.exists():
            raise FileNotFoundError(
                f"找不到 {path}。请把 Kaggle word2vec-nlp-tutorial 的三份 TSV 放进 "
                f"{CORPUS_DIR}/，或运行 tools/make_local_dataset.py 从 aclImdb 重建。"
            )
        frames.append(pd.read_csv(path, header=0, delimiter="\t", quoting=csv.QUOTE_NONE))
    train, test, unlabeled = frames
    logging.info("labeledTrainData %d 条 / testData %d 条 / unlabeledTrainData %d 条",
                 len(train), len(test), len(unlabeled))
    return train, test, unlabeled


def load_glove() -> KeyedVectors:
    for filename in GLOVE_CANDIDATES:
        path = GLOVE_DIR / filename
        if not path.exists():
            continue
        logging.info("加载词向量 %s ...", path)
        if path.suffix == ".kv":
            # mmap='r'：向量矩阵留在磁盘上按需读页，加载从分钟级降到秒级。
            return KeyedVectors.load(str(path), mmap="r")
        return KeyedVectors.load_word2vec_format(str(path), binary=False, no_header=True)
    raise FileNotFoundError(
        f"{GLOVE_DIR}/ 下找不到 {' 或 '.join(GLOVE_CANDIDATES)}。"
        "请先运行 python tools/prepare_glove.py"
    )


def pad_samples(features: list[list[int]], maxlen: int, pad: int = 0) -> list[list[int]]:
    """截断到 maxlen，不足的右侧补 PAD。原实现用 while 循环逐个 append，等价但慢。"""
    return [ids[:maxlen] + [pad] * max(0, maxlen - len(ids)) for ids in features]


def encode_samples(samples: list[list[str]], word_to_idx: dict[str, int]) -> list[list[int]]:
    """词 → id。未登录词映射到 0（<unk>，其向量为全零）。"""
    return [[word_to_idx.get(token, 0) for token in sample] for sample in samples]


def build_embedding(vectors: KeyedVectors, word_to_idx: dict[str, int],
                    embed_size: int) -> tuple[torch.Tensor, list[str]]:
    """构建 (vocab_size + 1, embed_size) 的 Embedding 矩阵。

    遍历方向和原代码相反：原代码遍历 GloVe 的 220 万词查 IMDB 词表，这里遍历
    IMDB 的约 10 万词查 GloVe——快 20 倍，而且能直接收集 OOV 列表。
    行 0 是 <unk>/<pad>，保持全零。
    """
    weight = torch.zeros(len(word_to_idx), embed_size)
    oov = []
    for word, index in word_to_idx.items():
        if index == 0:
            continue
        if word in vectors.key_to_index:
            # np.array(..., copy=True)：mmap 打开的 .npy 是只读的，torch.from_numpy
            # 直接包只读数组会报 non-writable tensor 警告。
            weight[index] = torch.from_numpy(np.array(vectors[word], dtype=np.float32))
        else:
            oov.append(word)
    return weight, oov


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=MAX_LEN, help="定长填充长度")
    parser.add_argument("--test-size", type=float, default=0.2, help="验证集占比")
    parser.add_argument("--output", type=Path, default=PICKLE_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(SEED)

    train, test, _ = read_corpus()

    logging.info("清洗与分词 ...")
    train_tokens = [review_to_wordlist(review) for review in train["review"]]
    test_tokens = [review_to_wordlist(review) for review in test["review"]]
    train_targets = train["sentiment"].tolist()

    lengths = np.array([len(tokens) for tokens in train_tokens])
    logging.info("训练集评论长度：均值 %.1f 词，中位数 %d，95 分位 %d，最长 %d",
                 lengths.mean(), np.median(lengths), np.percentile(lengths, 95), lengths.max())
    logging.info("max_len=%d 会截断 %.1f%% 的评论",
                 args.max_len, 100 * (lengths > args.max_len).mean())

    # 词表用「标注训练集 + 测试集」构建。不含 unlabeled 是刻意的：
    # 词表只需覆盖会被编码的文本，加进 5 万条无标注评论只会让 Embedding 矩阵变大。
    vocab = set(chain.from_iterable(train_tokens)) | set(chain.from_iterable(test_tokens))
    word_to_idx = {"<unk>": 0}
    word_to_idx.update({word: i + 1 for i, word in enumerate(sorted(vocab))})
    idx_to_word = {index: word for word, index in word_to_idx.items()}
    logging.info("词表大小 %d（含 <unk>，Embedding 矩阵行数 %d）", len(vocab), len(word_to_idx))

    # stratify 保证训练/验证集正负样本比例一致；原代码没加，划分会有随机偏斜。
    train_reviews, val_reviews, train_labels, val_labels = train_test_split(
        train_tokens, train_targets, test_size=args.test_size,
        random_state=SEED, stratify=train_targets,
    )
    logging.info("划分：训练 %d 条 / 验证 %d 条", len(train_reviews), len(val_reviews))

    logging.info("编码与定长填充（max_len=%d）...", args.max_len)
    train_features = torch.tensor(
        pad_samples(encode_samples(train_reviews, word_to_idx), args.max_len), dtype=torch.long)
    val_features = torch.tensor(
        pad_samples(encode_samples(val_reviews, word_to_idx), args.max_len), dtype=torch.long)
    test_features = torch.tensor(
        pad_samples(encode_samples(test_tokens, word_to_idx), args.max_len), dtype=torch.long)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    vectors = load_glove()
    logging.info("GloVe 词表 %d 词 × %d 维", len(vectors.index_to_key), vectors.vector_size)
    weight, oov = build_embedding(vectors, word_to_idx, EMBED_SIZE)

    logging.info("=== 数据形状 ===")
    for label, tensor in (("train_features", train_features), ("train_labels", train_labels),
                          ("val_features", val_features), ("val_labels", val_labels),
                          ("test_features", test_features), ("weight", weight)):
        logging.info("  %-15s %s %s", label, tuple(tensor.shape), tensor.dtype)

    logging.info("=== OOV ===")
    logging.info("  未命中 GloVe 的词型: %d / %d（%.2f%%）",
                 len(oov), len(vocab), 100 * len(oov) / len(vocab))
    # 词型 OOV 率会严重高估问题：OOV 几乎都是只出现一两次的拟声词、拼写错误和
    # 清洗残留碎片（`isn't` → `isn`）。按词次（token）算才反映真实覆盖率。
    oov_ids = torch.tensor([word_to_idx[word] for word in oov], dtype=torch.long)
    oov_mask = torch.zeros(len(word_to_idx), dtype=torch.bool)
    oov_mask[oov_ids] = True
    real_tokens = train_features != 0
    oov_tokens = oov_mask[train_features] & real_tokens
    logging.info("  未命中 GloVe 的词次: %d / %d（%.2f%%，训练集）",
                 int(oov_tokens.sum()), int(real_tokens.sum()),
                 100 * float(oov_tokens.sum()) / float(real_tokens.sum()))
    logging.info("  OOV 样例: %s", ", ".join(oov[:20]))
    covered_rows = int((weight.abs().sum(dim=1) > 0).sum())
    logging.info("  Embedding 非零行: %d / %d", covered_rows, weight.shape[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(
            [train_features, train_labels, val_features, val_labels, test_features,
             weight, word_to_idx, idx_to_word, vocab], handle, protocol=4)
    size_mb = args.output.stat().st_size / 1024 ** 2
    logging.info("已写出 %s（%.1f MB）", args.output, size_mb)


if __name__ == "__main__":
    main()
