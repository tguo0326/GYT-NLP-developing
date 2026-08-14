"""给**已有的**提交文件打分，不重新加载模型。

和 `score_test.py` 的分工：那个脚本从 checkpoint 重建模型再推理；这个脚本直接读
`results/<name>_submission.csv` 里的概率，对着公开标签算准确率和 AUC。

为什么需要这么一个脚本——这是阶段三踩到的一个**很隐蔽的坑**：

DeBERTa 的分类结构是 `encoder → pooler.dense (1024×1024) → classifier (1024→2)`，
而 `pooler.dense` 和 `classifier` 在 `from_pretrained` 时**都是随机初始化的**
（日志里那句 "newly initialized: classifier.*, pooler.dense.*"）。

PEFT 只把 `classifier` 放进 `modules_to_save`，`pooler` 既不训练也不保存。
训练时它就固定在那一份随机权重上，LoRA 和分类头是学着去配合**那个特定的随机投影**的。
重新加载时 `from_pretrained` 会生成**另一个**随机 pooler，学到的分类方向随即失效：

    验证集（训练进程内）   0.9566
    重新加载后再推理       0.4417   ROC-AUC 0.3116   ← 比瞎猜还差，反相关

**不报任何错**，只是数字不对。而训练进程内产出的那份 submission 是正确的——
它用的就是内存里那个完整模型。所以打分应当直接读那份 CSV。

    python tools/score_submissions.py --model peft
    python tools/score_submissions.py --model deberta_lora
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import common  # noqa: E402

import score_test  # noqa: E402

PEFT_MODELS = list(score_test.PEFT_MODELS)


def score_submission(name: str, test: pd.DataFrame, truth: pd.Series) -> dict | None:
    path = common.RESULTS_DIR / f"{name}_submission.csv"
    if not path.exists():
        print(f"\n=== {name} ===\n  跳过：找不到 {path.name}")
        return None

    submission = pd.read_csv(path, quoting=csv.QUOTE_NONE)
    # 官方格式的标题行是带引号的 `"id","sentiment"`，用 QUOTE_NONE 读进来列名就带引号；
    # 而 pandas 自己写出的那份没有引号。两种都要能读。
    submission.columns = [c.strip('"') for c in submission.columns]
    print(f"\n=== {name} ===")
    if len(submission) != len(test):
        print(f"  ⚠ 行数不符：{len(submission):,} vs 测试集 {len(test):,}，跳过")
        return None

    # 按 id 严格对齐，不靠行序。行序错位会让分数等于随机，
    # 而文件看起来完全正常——这是本项目在阶段二踩过的坑。
    merged = test[["id"]].merge(submission, on="id", how="left")
    if merged["sentiment"].isna().any():
        missing = int(merged["sentiment"].isna().sum())
        print(f"  ⚠ 有 {missing:,} 个 id 在提交文件里找不到，跳过")
        return None

    probabilities = merged["sentiment"].to_numpy(dtype=float)
    predictions = (probabilities >= 0.5).astype(int)

    mask = truth.notna().to_numpy()
    y_true = truth[mask].astype(int).to_numpy()
    row = {
        "model": name,
        "scored_rows": int(mask.sum()),
        "test_acc": round(float(accuracy_score(y_true, predictions[mask])), 4),
        "test_auc": round(float(roc_auc_score(y_true, probabilities[mask])), 4),
    }
    print(f"  测试集准确率 {row['test_acc']:.4f}   ROC-AUC {row['test_auc']:.4f}"
          f"   （在 {row['scored_rows']:,} / {len(test):,} 条有标签的行上）")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="peft",
                        help="模型名，或 peft（阶段三四种）/ all（results/ 下所有提交文件）")
    parser.add_argument("--merge", action="store_true", default=True,
                        help="把结果并入 results/test_scores.csv（已有的同名行会被替换）")
    args = parser.parse_args()

    test, truth = score_test.load_test_frame()
    if truth is None:
        raise SystemExit("没有 corpus/imdb/testDataWithLabels.tsv，无法本地打分")
    print(f"测试集 {len(test):,} 条，其中 {int(truth.notna().sum()):,} 条有公开标签")

    if args.model == "peft":
        names = PEFT_MODELS
    elif args.model == "all":
        names = sorted(p.name.removesuffix("_submission.csv")
                       for p in common.RESULTS_DIR.glob("*_submission.csv"))
    else:
        names = [args.model]

    rows = [row for row in (score_submission(n, test, truth) for n in names) if row]
    if not rows:
        raise SystemExit("没有任何可打分的提交文件")

    frame = pd.DataFrame(rows)
    path = common.RESULTS_DIR / "test_scores.csv"
    if args.merge and path.exists():
        existing = pd.read_csv(path)
        # 同名模型用新结果替换，其余保留——不能把阶段二已有的分数冲掉
        existing = existing[~existing["model"].isin(frame["model"])]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = frame.sort_values("test_auc", ascending=False)
    frame.to_csv(path, index=False)

    print("\n=== 测试集汇总（按 AUC 排序）===")
    print(frame.to_string(index=False))
    print(f"\n已写出 {path}")


if __name__ == "__main__":
    main()
