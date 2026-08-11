"""任务 4：把 Stanford GloVe 840B.300d 转成新版 Gensim 可直接加载的格式。

为什么不能直接 `KeyedVectors.load_word2vec_format(txt, no_header=True)`：

1. GloVe 的原始 txt 没有 `<词数> <维度>` 头部行，这一点 `no_header=True` 能解决；
2. 但 840B 这一份里有若干「词」本身包含空格或就是一个空格（例如 `. . .`、`at&t`
   之外的一些噪声条目）。Gensim 按空格切分后要求恰好 301 段，遇到这些行会直接
   抛 ValueError。所以这里自己解析：**从右边取 300 个数当向量，剩下的整段当词**，
   解析失败的行计数后跳过。

产物是 Gensim 原生格式 `glove/glove.840B.300d.kv`（+ 同名 .vectors.npy）。
相比每次重新读 5.6 GB 文本，原生格式支持 `mmap='r'`，加载从分钟级降到秒级。

    python tools/prepare_glove.py            # 解压 + 转换 + 自检
    python tools/prepare_glove.py --test-only  # 只跑自检
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

DIM = 300
ARCHIVE = "glove.840B.300d.zip"
TEXT = "glove.840B.300d.txt"
KV = "glove.840B.300d.kv"
DOWNLOAD_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"


def ensure_text(glove_dir: Path) -> Path:
    """确保 txt 存在；只有 zip 时自动解压。"""
    text_path = glove_dir / TEXT
    if text_path.exists():
        return text_path

    archive_path = glove_dir / ARCHIVE
    if not archive_path.exists():
        raise FileNotFoundError(
            f"缺少 {archive_path}。请先下载：\n  wget -c -O {archive_path} {DOWNLOAD_URL}"
        )
    print(f"解压 {archive_path} → {text_path}（约 5.6 GB，需要几分钟）")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extract(TEXT, glove_dir)
    return text_path


def convert(text_path: Path, kv_path: Path) -> None:
    """流式读入 txt，写出 Gensim 原生 KeyedVectors。"""
    print(f"解析 {text_path} …")
    words: list[str] = []
    rows: list[np.ndarray] = []
    skipped = 0

    with text_path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            parts = line.rstrip("\n").rstrip().split(" ")
            if len(parts) < DIM + 1:
                skipped += 1
                continue
            # 从右取 300 个数，左边剩下的合并成词——这样含空格的词也不会丢。
            word = " ".join(parts[: len(parts) - DIM])
            try:
                vector = np.asarray(parts[-DIM:], dtype=np.float32)
            except ValueError:
                skipped += 1
                continue
            words.append(word)
            rows.append(vector)
            if lineno % 250_000 == 0:
                print(f"  已读 {lineno:,} 行")

    print(f"共 {len(words):,} 个词，跳过 {skipped} 行异常")

    # 注意不要传 count=len(words)：那是「预分配」语义，add_vectors 会在预分配的
    # 空行之后再追加一遍，词表变成两倍大、一半是没有 key 的零向量，
    # most_similar 会因为对零向量求余弦而刷 invalid value in divide 警告。
    vectors = KeyedVectors(vector_size=DIM)
    # 一次性 add_vectors 比逐条 add 快一个数量级。
    vectors.add_vectors(words, np.vstack(rows))
    vectors.save(str(kv_path))
    print(f"已保存 {kv_path}")


def selftest(kv_path: Path) -> None:
    """任务 4 要求的三项验证：取向量、找近义词、算相似度。"""
    print(f"\n加载 {kv_path}（mmap 模式）…")
    vectors = KeyedVectors.load(str(kv_path), mmap="r")
    print(f"词表大小 {len(vectors.index_to_key):,}，维度 {vectors.vector_size}")

    print("\n--- 1. 单词向量 ---")
    vector = vectors["movie"]
    print(f"movie: shape={vector.shape} dtype={vector.dtype} 范数={np.linalg.norm(vector):.3f}")
    print(f"前 8 维: {np.round(vector[:8], 4).tolist()}")

    print("\n--- 2. 相似词 ---")
    for word in ("movie", "awful", "brilliant", "france"):
        neighbours = ", ".join(
            f"{token}({score:.3f})" for token, score in vectors.most_similar(word, topn=6)
        )
        print(f"{word:10s} → {neighbours}")

    print("\n--- 3. 单词相似度 ---")
    pairs = [
        ("good", "great"),      # 近义
        ("good", "bad"),        # 反义：分布相似，余弦仍然高——这是 GloVe 的固有特点
        ("movie", "film"),
        ("movie", "banana"),    # 无关
        ("king", "queen"),
    ]
    for left, right in pairs:
        print(f"{left:8s} ~ {right:8s} = {vectors.similarity(left, right):.4f}")

    print("\n--- 4. 向量算术 ---")
    result = vectors.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
    print("king - man + woman → " + ", ".join(f"{t}({s:.3f})" for t, s in result))

    print("\n✓ 任务 4 自检通过")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove-dir", type=Path, default=Path("glove"))
    parser.add_argument("--test-only", action="store_true", help="跳过转换，只跑自检")
    args = parser.parse_args()

    kv_path = args.glove_dir / KV
    if not args.test_only:
        if kv_path.exists():
            print(f"{kv_path} 已存在，跳过转换（删掉它可强制重建）")
        else:
            convert(ensure_text(args.glove_dir), kv_path)
    selftest(kv_path)


if __name__ == "__main__":
    main()
