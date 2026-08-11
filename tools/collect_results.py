"""任务 12：把 results/*_summary.json 汇总成统一对比表。

每个模型脚本跑完都会写一份 `results/<name>_summary.json`；这里读全部现存的
summary，按固定顺序排成 Markdown 表格，写入 `results/comparison.md`，
同时打印四项结论（最高准确率 / 最快 / 参数最少 / 综合最好）。

「综合最好」用的判据：在准确率距最高不超过 1 个百分点的模型里，取每秒训练时间
换来的准确率最高者——也就是「没有明显牺牲精度的前提下最省算力」。

    python tools/collect_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# 展示顺序与中文名。没跑的模型自动跳过。
ORDER = [
    ("bow_rf", "传统分类器（随机森林）"),
    ("tfidf_lr", "传统分类器（逻辑回归）"),
    ("cnn", "CNN"),
    ("lstm", "LSTM"),
    ("gru", "GRU"),
    ("cnnlstm", "CNN-LSTM"),
    ("attention_lstm", "Attention-LSTM"),
    ("transformer", "Transformer"),
    ("capsule_lstm", "Capsule-LSTM"),
    ("bert", "BERT"),
    ("distilbert", "DistilBERT"),
    ("roberta", "RoBERTa"),
]


def load_summaries() -> list[tuple[str, dict]]:
    rows = []
    for name, label in ORDER:
        path = RESULTS_DIR / f"{name}_summary.json"
        if path.exists():
            rows.append((label, json.loads(path.read_text(encoding="utf-8"))))
    return rows


def format_table(rows: list[tuple[str, dict]]) -> str:
    header = (
        "| 模型 | 文本表示 | 验证集准确率 | 训练时间 | 参数量 | 可训练参数 | 最佳 epoch |\n"
        "| --- | --- | --: | --: | --: | --: | --: |\n"
    )
    lines = []
    for label, summary in rows:
        epoch = summary.get("best_epoch")
        lines.append("| {} | {} | {:.4f} | {:.0f} s | {:,} | {:,} | {} |".format(
            label,
            summary["text_representation"],
            summary["best_val_acc"],
            summary.get("train_seconds") or 0,
            summary.get("total_params") or 0,
            summary.get("trainable_params") or 0,
            epoch if epoch is not None else "—",
        ))
    return header + "\n".join(lines) + "\n"


TRADITIONAL = {"bow_rf", "tfidf_lr"}


def _four_answers(rows: list[tuple[str, dict]]) -> list[str]:
    best_acc = max(rows, key=lambda item: item[1]["best_val_acc"])
    fastest = min(rows, key=lambda item: item[1].get("train_seconds") or float("inf"))
    smallest = min(rows, key=lambda item: item[1].get("trainable_params") or float("inf"))

    top = best_acc[1]["best_val_acc"]
    # 综合最好：准确率距最高 1 个点以内，训练时间最短
    contenders = [item for item in rows if item[1]["best_val_acc"] >= top - 0.01]
    overall = min(contenders, key=lambda item: item[1].get("train_seconds") or float("inf"))

    return [
        f"- **准确率最高**：{best_acc[0]}，{top:.4f}",
        f"- **训练最快**：{fastest[0]}，{fastest[1].get('train_seconds', 0):.0f} 秒",
        f"- **可训练参数最少**：{smallest[0]}，{smallest[1].get('trainable_params', 0):,}",
        f"- **综合最好**：{overall[0]}"
        f"（准确率 {overall[1]['best_val_acc']:.4f}，训练 {overall[1].get('train_seconds', 0):.0f} 秒——"
        f"在距最高准确率 1 个百分点以内的模型里训练时间最短）",
    ]


def conclusions(rows: list[tuple[str, dict]]) -> str:
    if not rows:
        return "（还没有任何结果）\n"

    neural = [item for item in rows if item[1]["model"] not in TRADITIONAL]
    lines = ["### 全部模型", ""] + _four_answers(rows)
    if neural and len(neural) != len(rows):
        # 传统稀疏基线在准确率和速度上都很能打，容易把神经网络之间的差异盖掉，
        # 所以再单列一份只看神经网络的结论。
        lines += ["", "### 只看神经网络模型", ""] + _four_answers(neural)
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_summaries()
    if not rows:
        raise SystemExit(f"{RESULTS_DIR} 下没有任何 *_summary.json，请先训练模型")

    device = next((s.get("device") for _, s in rows if s.get("device") != "cpu"), "cpu")
    body = "\n".join([
        "# 模型对比表",
        "",
        "所有神经网络模型共用同一份 `pickle/imdb_glove.pickle3`：",
        "20,000 条训练 / 5,000 条验证（8:2 分层划分，`random_state=42`），",
        "定长填充 512，Embedding 用 GloVe 840B.300d 初始化并**冻结**。",
        f"训练设备：{device}。传统分类器的「参数量」一列填的是特征维度。",
        "",
        format_table(rows),
        "## 结论",
        "",
        conclusions(rows),
    ])
    path = RESULTS_DIR / "comparison.md"
    path.write_text(body, encoding="utf-8")
    print(body)
    print(f"已写出 {path}")

    inject_into_readme(format_table(rows) + "\n" + conclusions(rows))


README_START = "<!-- RESULTS_TABLE_START -->"
README_END = "<!-- RESULTS_TABLE_END -->"


def inject_into_readme(block: str) -> None:
    """把表格同步进 README 的标记区间，避免两处数字手工维护会漂移。"""
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        print(f"README.md 里没有 {README_START} 标记，跳过同步")
        return
    head, _, rest = text.partition(README_START)
    _, _, tail = rest.partition(README_END)
    readme.write_text(f"{head}{README_START}\n\n{block}\n{README_END}{tail}", encoding="utf-8")
    print(f"已同步表格进 {readme.name}")


if __name__ == "__main__":
    main()
