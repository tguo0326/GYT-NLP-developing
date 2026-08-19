"""画 loss 曲线与验证指标曲线。

    python plot_curves.py            # 汇总 result/ 下所有正式实验
输出 result/curves_bs<N>.png：左 = CE / 对比 / total loss，中 = 每 query 候选与正样本数，
右 = 验证集 accuracy / ROC-AUC 随 epoch 的变化。
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results"
LABEL = {"baseline": "Baseline", "scl": "SCL", "scl_moco": "SCL-MoCo"}
COLOR = {"baseline": "#4E79A7", "scl": "#F28E2B", "scl_moco": "#59A14F"}
COLOR_M0999 = "#B07AA1"   # m=0.999 那轮单独用一个颜色


def read_steps(tag):
    path = RESULT / f"{tag}_steps.csv"
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    runs = defaultdict(list)
    for path in sorted(RESULT.glob("scl_moco_*_summary.json")):
        s = json.loads(path.read_text())
        if s["config"].get("subset") or s["config"].get("probe_steps"):
            continue
        runs[s["config"]["batch_size"]].append(s)
    if not runs:
        print("没有正式实验结果可画")
        return

    for batch_size, group in sorted(runs.items()):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for s in sorted(group, key=lambda x: x["method"]):
            method, color = s["method"], COLOR[s["method"]]
            if method == "scl_moco" and s["config"]["momentum"] > 0.995:
                color = COLOR_M0999
            name = LABEL[method] + (f" m={s['config']['momentum']:g}"
                                    if method == "scl_moco" else "")
            steps = read_steps(s["tag"])
            if steps:
                x = [int(r["micro_step"]) for r in steps]
                axes[0].plot(x, [float(r["ce_loss"]) for r in steps], color=color,
                             label=f"{name} CE")
                if method != "baseline":
                    axes[0].plot(x, [float(r["contrastive_loss"]) for r in steps],
                                 color=color, ls="--",
                                 label=f"{name} contrastive")
                    axes[1].plot(x, [float(r["n_candidates"]) for r in steps],
                                 color=color, label=f"{name} candidates")
                    axes[1].plot(x, [float(r["pos_per_query"]) for r in steps],
                                 color=color, ls=":", label=f"{name} positives")
            per_epoch = s.get("val_per_epoch") or []
            if per_epoch:
                ep = [e["epoch"] for e in per_epoch]
                axes[2].plot(ep, [e["eval_accuracy"] for e in per_epoch], marker="o",
                             color=color, label=f"{name} val acc")
                axes[2].plot(ep, [e["eval_roc_auc"] for e in per_epoch], marker="s",
                             ls="--", color=color, label=f"{name} val AUC")

        axes[0].set(title=f"Loss (real batch={batch_size})", xlabel="micro-batch",
                    ylabel="loss")
        axes[1].set(title="Candidates / positives per query", xlabel="micro-batch",
                    ylabel="count", yscale="log")
        axes[2].set(title="Validation metrics", xlabel="epoch", ylabel="score")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        out = RESULT / f"scl_moco_curves_bs{batch_size}.png"
        fig.savefig(out, dpi=140)
        print(f"已写出 {out}")


if __name__ == "__main__":
    main()
