"""任务 3 的数据体检：确认 corpus/imdb/ 下三份 TSV 真的能用。

单纯 `read_csv` 不报错并不代表数据是对的——用 QUOTE_NONE 之外的规则解析这份
TSV 时，影评正文里的引号会把字段吞掉，结果是行数正常、内容错位。所以这里逐项
检查行数、字段名、空值、标签取值，并抽样打印评论与标签，人眼确认二者对应。

    python tools/check_imdb_data.py [--data-dir corpus/imdb]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

# 期望的行数与字段：Kaggle word2vec-nlp-tutorial 的官方切分。
EXPECTED = {
    "labeledTrainData": (25_000, ["id", "sentiment", "review"]),
    "testData": (25_000, ["id", "review"]),
    "unlabeledTrainData": (50_000, ["id", "review"]),
}


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)


def check(frame: pd.DataFrame, stem: str) -> list[str]:
    """返回该文件的问题列表；空列表表示全部通过。"""
    problems = []
    expected_rows, expected_cols = EXPECTED[stem]

    if len(frame) != expected_rows:
        problems.append(f"行数 {len(frame):,}，期望 {expected_rows:,}")
    if list(frame.columns) != expected_cols:
        problems.append(f"字段 {list(frame.columns)}，期望 {expected_cols}")

    null_counts = frame.isnull().sum()
    if null_counts.any():
        problems.append(f"存在空值：{null_counts[null_counts > 0].to_dict()}")

    empty = (frame["review"].astype(str).str.strip() == "").sum()
    if empty:
        problems.append(f"{empty} 条 review 是空字符串")

    if frame["id"].duplicated().any():
        problems.append(f"{frame['id'].duplicated().sum()} 个 id 重复")

    if "sentiment" in frame.columns:
        values = sorted(frame["sentiment"].unique().tolist())
        if values != [0, 1]:
            problems.append(f"sentiment 取值 {values}，期望 [0, 1]")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("corpus/imdb"))
    args = parser.parse_args()

    all_ok = True
    frames = {}
    for stem in EXPECTED:
        path = args.data_dir / f"{stem}.tsv"
        print(f"\n=== {path} ===")
        if not path.exists():
            print("  ✗ 文件不存在")
            all_ok = False
            continue

        frame = load(path)
        frames[stem] = frame
        lengths = frame["review"].astype(str).str.split().str.len()
        print(f"  行数     : {len(frame):,}")
        print(f"  字段     : {list(frame.columns)}")
        print(f"  评论长度 : 均值 {lengths.mean():.1f} 词，"
              f"中位数 {lengths.median():.0f}，最长 {lengths.max()}，最短 {lengths.min()}")
        if "sentiment" in frame.columns:
            counts = frame["sentiment"].value_counts().sort_index()
            print(f"  标签分布 : 负面 {counts[0]:,} / 正面 {counts[1]:,}")

        problems = check(frame, stem)
        if problems:
            all_ok = False
            for item in problems:
                print(f"  ✗ {item}")
        else:
            print("  ✓ 行数、字段、空值、id 唯一性、标签取值全部正常")

    # 评论与标签是否对应：抽样人眼确认，比任何统计量都直接。
    if "labeledTrainData" in frames:
        print("\n=== 抽样确认评论与标签对应 ===")
        sample = frames["labeledTrainData"].sample(4, random_state=42)
        for row in sample.itertuples():
            tag = "正面" if row.sentiment == 1 else "负面"
            print(f"\n  [{row.id}] 标签={row.sentiment}（{tag}）")
            print(f"  {row.review[:220]} ...")

    print("\n" + ("✓ 任务 3 数据检查通过" if all_ok else "✗ 存在问题，见上"))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
