"""三组两两做 McNemar 检验：差异到底是不是随机波动。

Accuracy 的差值看着小（0.08 个百分点），但「小」不等于「不显著」——
要看的是**同一批样本上两个模型的分歧结构**：McNemar 只用两个模型判断不一致的那些样本，
比直接比 accuracy 的两个独立二项分布更有力。

    python significance.py
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import data

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "Baseline": "scl_moco_baseline_bs4_seed42",
    "SCL": "scl_moco_scl_bs4_seed42",
    "SCL-MoCo m=0.999": "scl_moco_m0999_bs4_seed42",
    "SCL-MoCo m=0.99": "scl_moco_m099_bs4_seed42",
}


def main():
    _, test_frame = data.load_frames()
    truth = data.load_test_labels(test_frame)
    mask = truth.notna().to_numpy()
    y = truth[mask].astype(int).to_numpy()

    correct, probs = {}, {}
    for name, tag in RUNS.items():
        sub = pd.read_csv(ROOT / "results" / f"{tag}_submission.csv",
                          quoting=csv.QUOTE_NONE)
        sub.columns = [c.strip('"') for c in sub.columns]
        merged = test_frame[["id"]].merge(sub, on="id", how="left")
        p = merged["sentiment"].to_numpy(dtype=float)[mask]
        probs[name] = p
        correct[name] = (p >= 0.5).astype(int) == y

    n = len(y)
    se = np.sqrt(0.96 * 0.04 / n)
    print(f"打分样本 {n} 条；accuracy≈0.96 时单模型的二项标准误 = {se * 100:.3f} 个百分点")
    print(f"→ 1 条样本 = {100 / n:.4f} 个百分点\n")

    print("| 对比 | 各自 acc | 差值(百分点) | 只有前者对 | 只有后者对 | McNemar p |")
    print("|---|---|---|---|---|---|")
    for a, b in combinations(RUNS, 2):
        ca, cb = correct[a], correct[b]
        only_a = int((ca & ~cb).sum())
        only_b = int((~ca & cb).sum())
        # 精确二项检验（McNemar 的 exact 版本，分歧样本数不大时比卡方更稳）
        p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue
        print(f"| {a} vs {b} | {ca.mean():.4f} / {cb.mean():.4f} "
              f"| {100 * (ca.mean() - cb.mean()):+.3f} | {only_a} | {only_b} "
              f"| {p:.3f} |")

    print("\n### 概率层面的相关性（模型之间到底有多像）\n")
    print("| 对比 | 预测概率 Pearson r | 硬标签一致率 |")
    print("|---|---|---|")
    for a, b in combinations(RUNS, 2):
        r = np.corrcoef(probs[a], probs[b])[0, 1]
        agree = ((probs[a] >= 0.5) == (probs[b] >= 0.5)).mean()
        print(f"| {a} vs {b} | {r:.4f} | {agree:.4f} |")


if __name__ == "__main__":
    main()
