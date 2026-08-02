"""Rebuild the competition-style TSV files from the original Stanford IMDB corpus.

Kaggle 的 `word2vec-nlp-tutorial` 数据本身就是 Stanford aclImdb 语料的一个切分。
竞赛文件需要登录 Kaggle 才能下载，所以本脚本提供一条离线路径：直接从公开的
aclImdb 原始语料生成同样列结构的 TSV，方便在本地跑通、调试和回归测试。

    labeledTrainData.tsv     id, sentiment, review   (25,000 条，来自 aclImdb/train)
    testData.tsv             id, review              (25,000 条，来自 aclImdb/test)
    unlabeledTrainData.tsv   id, review              (50,000 条，来自 aclImdb/train/unsup)

注意：这里重建的 `id` 沿用原始文件名（形如 `12345_9`），与 Kaggle 官方文件的行
顺序和 id 编号不完全一致。要提交到排行榜，请使用 Kaggle 挂载的官方数据。

用法：

    python tools/make_local_dataset.py --imdb-dir /path/to/aclImdb --output-dir data
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

# TSV 用制表符分隔、且以 QUOTE_NONE 读取，因此正文里的制表符和换行必须先压平。
WHITESPACE = re.compile(r"\s+")


def read_split(split_dir: Path, labels: dict[str, int] | None) -> pd.DataFrame:
    """把 aclImdb 的一个目录读成 DataFrame。

    labels 为 None 时表示无标签数据（测试集/unsup），只返回 id 与 review。
    """
    rows = []
    subdirs = sorted(labels) if labels else ["unsup"]
    for subdir in subdirs:
        for path in sorted((split_dir / subdir).glob("*.txt")):
            text = WHITESPACE.sub(" ", path.read_text(encoding="utf-8")).strip()
            row = {"id": path.stem, "review": text}
            if labels:
                row["sentiment"] = labels[subdir]
            rows.append(row)

    frame = pd.DataFrame(rows)
    columns = ["id", "sentiment", "review"] if labels else ["id", "review"]
    return frame[columns]


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar=None)
    print(f"{path}  ({len(frame):,} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imdb-dir", type=Path, required=True, help="解压后的 aclImdb 目录")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--skip-unlabeled", action="store_true", help="不生成 50,000 条无标签数据")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sentiment_labels = {"neg": 0, "pos": 1}

    write_tsv(
        read_split(args.imdb_dir / "train", sentiment_labels),
        args.output_dir / "labeledTrainData.tsv",
    )
    test = read_split(args.imdb_dir / "test", sentiment_labels)
    # 竞赛的 testData.tsv 不含标签；另存一份带标签的副本便于本地离线评估。
    write_tsv(test[["id", "review"]], args.output_dir / "testData.tsv")
    write_tsv(test, args.output_dir / "testDataWithLabels.tsv")

    if not args.skip_unlabeled:
        write_tsv(
            read_split(args.imdb_dir / "train", None),
            args.output_dir / "unlabeledTrainData.tsv",
        )


if __name__ == "__main__":
    main()
