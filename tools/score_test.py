"""在 25,000 条测试集上打分，并生成 Kaggle 可提交的概率文件。

两个用途：

1. **本地算真实测试集分数**。`corpus/imdb/testDataWithLabels.tsv` 是从 Stanford
   aclImdb 重建的——**aclImdb 的测试集标签是公开的**，所以不用等 Kaggle 排行榜，
   本地就能算出准确率和 AUC。这是比验证集更硬的数字（模型从没见过这 25,000 条）。
2. **生成 Kaggle 提交文件**。这里输出的是**正面概率**而不是 0/1 硬标签——
   竞赛的评价指标是 ROC-AUC，交硬标签相当于把概率信息全丢掉，
   实测能差 3~5 个百分点的 AUC。

    python tools/score_test.py --model roberta
    python tools/score_test.py --model all
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402

# GloVe 系模型：脚本模块名 → 权重文件名（common.train 存的是 models/<name>_best.pt）
GLOVE_MODELS = {
    "cnn": "imdb_cnn",
    "lstm": "imdb_lstm",
    "gru": "imdb_gru",
    "cnnlstm": "imdb_cnnlstm",
    "attention_lstm": "imdb_attention_lstm",
    "transformer": "imdb_transformer",
    "capsule_lstm": "imdb_capsule_lstm",
}
HF_MODELS = ("distilbert", "bert", "roberta")


def load_test_frame() -> tuple[pd.DataFrame, pd.Series | None]:
    """读测试集。有 testDataWithLabels.tsv 就顺带返回真实标签。"""
    test = pd.read_csv(common.CORPUS_DIR / "testData.tsv", header=0,
                       delimiter="\t", quoting=csv.QUOTE_NONE)
    labelled_path = common.CORPUS_DIR / "testDataWithLabels.tsv"
    if not labelled_path.exists():
        return test, None

    labelled = pd.read_csv(labelled_path, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
    # 按 id 对齐，不依赖行顺序——换成 Kaggle 官方 testData.tsv 后顺序会不同
    merged = test[["id"]].merge(labelled[["id", "sentiment"]], on="id", how="left")
    if merged["sentiment"].isna().any():
        missing = int(merged["sentiment"].isna().sum())
        print(f"⚠ {missing} 条 id 在 testDataWithLabels.tsv 里找不到，跳过本地打分。"
              "（用 Kaggle 官方 testData.tsv 时这是正常的）")
        return test, None
    return test, merged["sentiment"].astype(int)


def glove_probabilities(name: str, bundle: common.Bundle, test: pd.DataFrame,
                        device: torch.device) -> np.ndarray:
    module = importlib.import_module(GLOVE_MODELS[name])
    model = module.SentimentNet(bundle.weight)
    common.load_best(name, model, device)

    # pickle 里的 test_features 是按当前 corpus/imdb/testData.tsv 编码的。
    # 行数一致就直接用，否则重新编码（换成 Kaggle 官方文件的情况）。
    if len(bundle.test_features) == len(test):
        features = bundle.test_features
    else:
        print("  测试集行数与 pickle 不一致，按当前 TSV 重新编码")
        features = common.encode_texts(test["review"].tolist(), bundle.word_to_idx)

    logits = common.predict_logits(model, features, device)
    return torch.softmax(logits, dim=1)[:, 1].numpy()


def hf_probabilities(name: str, test: pd.DataFrame, device: torch.device,
                     batch_size: int = 64, max_length: int = 256) -> np.ndarray:
    import hf_trainer  # noqa: F401  —— 顺带关掉 transformers 的 TF 探测
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint = common.MODELS_DIR / f"{name}_hf"
    if not checkpoint.exists():
        raise FileNotFoundError(f"找不到 {checkpoint}，请先跑 python imdb_{name}_trainer.py")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).to(device).eval()

    texts = test["review"].astype(str).tolist()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], truncation=True,
                              max_length=max_length, padding=True, return_tensors="pt").to(device)
            chunks.append(torch.softmax(model(**batch).logits, dim=1)[:, 1].cpu())
    return torch.cat(chunks).numpy()


def score_one(name: str, bundle: common.Bundle | None, test: pd.DataFrame,
              truth: pd.Series | None, device: torch.device) -> dict:
    print(f"\n=== {name} ===")
    if name in HF_MODELS:
        probabilities = hf_probabilities(name, test, device)
    else:
        probabilities = glove_probabilities(name, bundle, test, device)

    predictions = (probabilities >= 0.5).astype(int)
    row = {"model": name}

    if truth is not None:
        row["test_acc"] = round(float(accuracy_score(truth, predictions)), 4)
        row["test_auc"] = round(float(roc_auc_score(truth, probabilities)), 4)
        print(f"  测试集准确率 {row['test_acc']:.4f}   ROC-AUC {row['test_auc']:.4f}")
    else:
        print("  （无标签，只生成提交文件）")

    # Kaggle 的指标是 AUC，所以提交概率而不是 0/1
    path = common.RESULTS_DIR / f"{name}_submission_proba.csv"
    pd.DataFrame({"id": test["id"], "sentiment": probabilities}).to_csv(
        path, index=False, quoting=csv.QUOTE_NONE)
    print(f"  已写出 {path.name}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="roberta",
                        help="模型名，或 all（跑全部已训练的模型）")
    args = parser.parse_args()

    device = common.get_device()
    print(f"设备：{device}")
    test, truth = load_test_frame()
    print(f"测试集 {len(test):,} 条" + ("，含真实标签" if truth is not None else "，无标签"))

    names = (list(GLOVE_MODELS) + list(HF_MODELS)) if args.model == "all" else [args.model]
    rows = []
    for name in names:
        try:
            rows.append(score_one(name, common.load_data() if name not in HF_MODELS else None,
                                  test, truth, device))
        except FileNotFoundError as error:
            print(f"\n=== {name} ===\n  跳过：{error}")

    if truth is not None and rows:
        frame = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
        path = common.RESULTS_DIR / "test_scores.csv"
        frame.to_csv(path, index=False)
        print("\n=== 测试集汇总（按 AUC 排序）===")
        print(frame.to_string(index=False))
        print(f"\n已写出 {path}")


if __name__ == "__main__":
    main()
