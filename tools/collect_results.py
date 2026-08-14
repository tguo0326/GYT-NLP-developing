"""把 results/ 下的分数汇总成对比表，写进 results/comparison.md 并同步到 README。

口径：**只报测试集**。25,000 条官方 testData.tsv，其中 24,961 条能与 aclImdb
的公开标签按正文对齐，指标就在这些行上算。验证集只用来选 epoch，不进表——
两套数字并列容易让人分不清哪个是结论。

    python tools/collect_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# (summary 名, 展示名, 阶段, 文本表示)
MODELS = [
    ("bow_rf",          "随机森林",         1, "Bag of Words（5,000 词频）"),
    ("tfidf_lr",        "逻辑回归",         1, "TF-IDF（1-2gram, 200,000）"),
    ("cnn",             "TextCNN",          2, "GloVe 840B.300d"),
    ("lstm",            "BiLSTM",           2, "GloVe 840B.300d"),
    ("gru",             "BiGRU",            2, "GloVe 840B.300d"),
    ("cnnlstm",         "CNN-LSTM",         2, "GloVe 840B.300d"),
    ("attention_lstm",  "Attention-LSTM",   2, "GloVe 840B.300d"),
    ("transformer",     "Transformer",      2, "GloVe 840B.300d"),
    ("capsule_lstm",    "Capsule-LSTM",     2, "GloVe 840B.300d"),
    ("distilbert",      "DistilBERT",       2, "全量微调"),
    ("bert",            "BERT",             2, "全量微调"),
    ("roberta",         "RoBERTa",          2, "全量微调"),
    ("deberta_prefix",  "Prefix-Tuning",    3, "RoBERTa-base，冻结底座"),
    ("deberta_ptuning", "P-Tuning",         3, "DeBERTa-v3-large，冻结底座"),
    ("deberta_adalora", "AdaLoRA",          3, "DeBERTa-v3-large，冻结底座"),
    ("deberta_lora",    "LoRA",             3, "DeBERTa-v3-large，冻结底座"),
]

PEFT = {"deberta_lora", "deberta_adalora", "deberta_ptuning", "deberta_prefix"}
FULL_FINETUNE = {"bert", "distilbert", "roberta"}


def load() -> list[dict]:
    scores = {}
    path = RESULTS_DIR / "test_scores.csv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                scores[row["model"]] = row

    rows = []
    for name, label, stage, representation in MODELS:
        summary_path = RESULTS_DIR / f"{name}_summary.json"
        if not summary_path.exists() or name not in scores:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "name": name, "label": label, "stage": stage,
            "representation": representation,
            "acc": float(scores[name]["test_acc"]),
            "auc": float(scores[name]["test_auc"]),
            "trainable": int(summary.get("trainable_params") or 0),
            "total": int(summary.get("total_params") or 0),
            "seconds": float(summary.get("train_seconds") or 0),
            "peak_gpu": summary.get("peak_gpu_gb"),
            "trainable_pct": summary.get("trainable_pct"),
            "adapter_mb": summary.get("adapter_mb"),
        })
    return rows


def main_table(rows: list[dict]) -> str:
    lines = ["| 模型 | 文本表示 | 测试集准确率 | ROC-AUC | 可训练参数 | 训练时间 |",
             "| --- | --- | --: | --: | --: | --: |"]
    for row in sorted(rows, key=lambda r: r["auc"], reverse=True):
        seconds = f"{row['seconds']:.0f} s" if row["seconds"] else "—"
        lines.append(f"| {row['label']} | {row['representation']} | {row['acc']:.4f} | "
                     f"{row['auc']:.4f} | {row['trainable']:,} | {seconds} |")
    return "\n".join(lines) + "\n"


def peft_table(rows: list[dict]) -> str:
    peft = [r for r in rows if r["name"] in PEFT]
    if not peft:
        return ""
    lines = ["| 方法 | 测试集准确率 | ROC-AUC | 可训练参数 | 占全模型 | 显存峰值 | adapter |",
             "| --- | --: | --: | --: | --: | --: | --: |"]
    for row in sorted(peft, key=lambda r: r["auc"], reverse=True):
        lines.append(
            f"| {row['label']} | {row['acc']:.4f} | {row['auc']:.4f} | "
            f"{row['trainable']:,} | {row['trainable_pct']:.2f}% | "
            f"{row['peak_gpu']:.2f} GB | {row['adapter_mb']:.0f} MB |")
    return "\n".join(lines) + "\n"


def headline(rows: list[dict]) -> str:
    best = max(rows, key=lambda r: r["auc"])
    full = [r for r in rows if r["name"] in FULL_FINETUNE]
    if not full:
        return ""
    best_full = max(full, key=lambda r: r["auc"])
    gap = (best["acc"] - best_full["acc"]) * 100
    ratio = best_full["trainable"] / max(best["trainable"], 1)
    cheapest = min((r for r in rows if r["name"] in PEFT), key=lambda r: r["trainable"])
    return (
        f"- 最好的是 **{best['label']}**：准确率 {best['acc']:.4f}，AUC {best['auc']:.4f}\n"
        f"- 比最好的全量微调（{best_full['label']} {best_full['acc']:.4f}）"
        f"高 {gap:.2f} 个百分点，可训练参数只有它的 1/{ratio:.0f}\n"
        f"- 最省的是 **{cheapest['label']}**："
        f"{cheapest['trainable']:,} 个参数（{cheapest['trainable_pct']:.2f}%）"
        f"拿到 {cheapest['acc']:.4f}\n"
    )


MARKERS = [("<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->"),
           ("<!-- PEFT_TABLE_START -->", "<!-- PEFT_TABLE_END -->")]


def inject(block: str, start: str, end: str) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if start not in text or end not in text:
        print(f"README.md 里没有 {start}，跳过")
        return
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    readme.write_text(f"{head}{start}\n\n{block}\n{end}{tail}", encoding="utf-8")
    print(f"已同步 {start}")


def main() -> None:
    rows = load()
    if not rows:
        raise SystemExit("results/ 下没有可汇总的结果，先训练模型并跑 tools/score_submissions.py")

    body = "\n".join([
        "# 对比表",
        "",
        "统一口径：25,000 条标注影评按 8:2 分层划分（`random_state=42`）训练，",
        "在官方 `testData.tsv` 的 25,000 条上评测（24,961 条能与 aclImdb 公开标签对齐）。",
        "单卡 Tesla T4。表由 `python tools/collect_results.py` 生成。",
        "",
        main_table(rows),
        "## 结论",
        "",
        headline(rows),
        "## 参数高效微调",
        "",
        peft_table(rows),
    ])
    (RESULTS_DIR / "comparison.md").write_text(body, encoding="utf-8")
    print(body)

    inject(main_table(rows) + "\n" + headline(rows), *MARKERS[0])
    inject(peft_table(rows), *MARKERS[1])


if __name__ == "__main__":
    main()
