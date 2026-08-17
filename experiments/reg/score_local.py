"""用本地 testDataWithLabels.tsv 给 result/*.csv 打分，按实验组分别汇总。

注意：这是 Kaggle 那份带标签的测试集（社区流传版本），方便本地立刻看到 test acc，
不等于官方 leaderboard 分数。正式汇报仍应以 Kaggle submission 的分数为准，
两者一般只差小数点后第三位。
"""
import glob
import json
import os

import pandas as pd

import data

LABEL_PATH = os.path.join(data.DEFAULT_DATA_DIR, "testDataWithLabels.tsv")
SUBMISSION_DIR = data.SUBMISSION_DIR
ORDER = {"none": 0, "rdrop": 1, "scl": 2, "both": 3}
DISPLAY = {"none": "baseline", "rdrop": "+ R-Drop",
           "scl": "+ SCL", "both": "+ R-Drop + SCL"}


def strip_quotes(series):
    """quoting=3 读出来的 id 带字面引号，两边都剥掉才能 merge。"""
    return series.astype(str).str.strip('"')


def group_of(path, meta):
    """按 (骨干, 微调方式/后端) 把结果分组，路线① 和路线② 不混在一张表里。"""
    model = meta["args"].get("model_name", "?").split("/")[-1]
    if "_unsloth" in path:
        return f"{model} + LoRA (unsloth)"
    if "lora" in path:
        return f"{model} + LoRA (peft)"
    return f"{model} 全量微调 (路线①改 forward)"


def main():
    gold = pd.read_csv(LABEL_PATH, header=0, delimiter="\t", quoting=3)
    gold = gold[["id", "sentiment"]].rename(columns={"sentiment": "gold"})
    gold["id"] = strip_quotes(gold["id"])

    groups = {}
    for path in sorted(glob.glob(os.path.join(SUBMISSION_DIR, "*_metrics.json"))):
        meta = json.load(open(path))
        if meta["args"].get("limit") is not None:
            print(f"跳过 {path}（limit={meta['args']['limit']}，是 smoke test）")
            continue
        reg = meta.get("reg") or meta["args"].get("reg")
        pred = pd.read_csv(path.replace("_metrics.json", ".csv"),
                           quoting=3).rename(columns={"sentiment": "pred"})
        pred["id"] = strip_quotes(pred["id"])
        merged = gold.merge(pred, on="id", validate="one_to_one")
        if len(merged) != len(gold):
            print(f"跳过 {path}（id 对不上，只匹配到 {len(merged)}/{len(gold)} 条）")
            continue
        groups.setdefault(group_of(path, meta), []).append({
            "reg": reg,
            "val_acc": meta["val_metrics"]["eval_accuracy"],
            "test_acc": (merged.gold == merged.pred).mean(),
        })

    for name, rows in groups.items():
        rows.sort(key=lambda r: ORDER.get(r["reg"], 9))
        base = next((r for r in rows if r["reg"] == "none"), None)
        print(f"\n### {name}")
        print("| setting | val acc | test acc | Δtest vs baseline |")
        print("|---|---|---|---|")
        for r in rows:
            if base is None or r["reg"] == "none":
                delta = "—"
            else:
                delta = f"{r['test_acc'] - base['test_acc']:+.4f}"
            print(f"| {DISPLAY.get(r['reg'], r['reg'])} | {r['val_acc']:.4f} | "
                  f"{r['test_acc']:.4f} | {delta} |")


if __name__ == "__main__":
    main()
