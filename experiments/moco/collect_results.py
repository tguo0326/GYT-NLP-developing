"""汇总 result/*_summary.json，打印六组对照表（缺的组就是缺的，不填估计值）。

    python collect_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL = {"baseline": "Baseline", "scl": "SCL", "scl_moco": "SCL-MoCo"}
ORDER = {"baseline": 0, "scl": 1, "scl_moco": 2}


def label_of(summary):
    """SCL-MoCo 有多个动量系数的运行，标签里带上 m，否则两行看不出区别。"""
    name = LABEL[summary["method"]]
    if summary["method"] == "scl_moco":
        name += f" m={summary['config']['momentum']:g}"
    return name


def load():
    rows = []
    for path in sorted((ROOT / "results").glob("scl_moco_*_summary.json")):
        s = json.loads(path.read_text())
        cfg = s["config"]
        if cfg.get("subset") or cfg.get("probe_steps"):
            print(f"跳过 {path.name}（subset/probe，冒烟测试）")
            continue
        rows.append(s)
    return rows


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    rows = load()
    if not rows:
        print("result/ 下还没有正式实验结果")
        return
    rows.sort(key=lambda s: (s["config"]["batch_size"], ORDER[s["method"]],
                         -s["config"]["momentum"]))

    print("\n### 测试集（本地公开标签，与 0.9633 同一口径）\n")
    head = ("| 真实batch | 方法 | Accuracy | Macro-F1 | Precision | Recall | ROC-AUC "
            "| 平均正样本数 | 平均负样本数 | 峰值显存 | 时间 |")
    print(head)
    print("|" + "---|" * 11)
    for s in rows:
        t = s.get("test_metrics") or {}
        pos = s.get("mean_pos_per_query")
        neg = s.get("mean_neg_per_query")
        print(f"| {s['config']['batch_size']} | {label_of(s)} "
              f"| {fmt(t.get('accuracy'))} | {fmt(t.get('macro_f1'))} "
              f"| {fmt(t.get('macro_precision'))} | {fmt(t.get('macro_recall'))} "
              f"| {fmt(t.get('roc_auc'))} "
              f"| {'—' if pos is None else f'{pos:.1f}'} "
              f"| {'—' if neg is None else f'{neg:.1f}'} "
              f"| {s['peak_gpu_gb']:.2f} GB | {s['train_seconds'] / 60:.0f} min |")

    print("\n### 验证集（最后一个 epoch，与训练进程同一份权重）\n")
    print("| 真实batch | 方法 | Accuracy | Macro-F1 | ROC-AUC | 最佳epoch(val acc) "
          "| 最终CE | 最终对比loss | optimizer步数 | EMA次数 |")
    print("|" + "---|" * 10)
    for s in rows:
        v = s.get("val_metrics_last_epoch") or {}
        print(f"| {s['config']['batch_size']} | {label_of(s)} "
              f"| {fmt((v.get('accuracy') or v.get('eval_accuracy')))} | {fmt((v.get('macro_f1') or v.get('eval_macro_f1')))} "
              f"| {fmt((v.get('roc_auc') or v.get('eval_roc_auc')))} "
              f"| {s.get('val_best_epoch')} ({fmt(s.get('val_best_accuracy'))}) "
              f"| {fmt(s.get('final_ce_loss'))} | {fmt(s.get('final_contrastive_loss'))} "
              f"| {s.get('optimizer_steps')} | {s.get('ema_calls')} |")

    print("\n### 队列 / 候选规模 / 动量分支诊断\n")
    print("| 真实batch | 方法 | 每query候选数 | 队列有效长度 | 队列标签分布 | q·k | 对比目标完成度 |")
    print("|" + "---|" * 7)
    for s in rows:
        cand = s.get("mean_candidates_per_query")
        cos, prog = s.get("final_query_key_cosine"), s.get("final_contrast_progress")
        print(f"| {s['config']['batch_size']} | {label_of(s)} "
              f"| {'—' if cand is None else f'{cand:.0f}'} "
              f"| {s.get('queue_valid_end') or '—'} "
              f"| {s.get('queue_label_counts_end') or '—'} "
              f"| {'—' if cos is None else f'{cos:.3f}'} "
              f"| {'—' if prog is None else f'{prog:.0%}'} |")

    missing = {(bs, m) for bs in (4, 16) for m in LABEL} - \
              {(s["config"]["batch_size"], s["method"]) for s in rows}
    if missing:
        print("\n缺失的组（没跑就是没跑，不填估计值）：",
              ", ".join(f"bs{bs}-{LABEL[m]}" for bs, m in sorted(missing)))


if __name__ == "__main__":
    main()
