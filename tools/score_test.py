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

阶段三的 PEFT 模型请用 tools/score_submissions.py，原因见 docs/peft-lora.md。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import common  # noqa: E402

# GloVe 系模型：脚本模块名 → 权重文件名（common.train 存的是 models/<name>_best.pt）
GLOVE_MODELS = {
    "cnn": "experiments.glove.cnn",
    "lstm": "experiments.glove.lstm",
    "gru": "experiments.glove.gru",
    "cnnlstm": "experiments.glove.cnnlstm",
    "attention_lstm": "experiments.glove.attention_lstm",
    "transformer": "experiments.glove.transformer",
    "capsule_lstm": "experiments.glove.capsule_lstm",
}
HF_MODELS = ("distilbert", "bert", "roberta")

# 阶段三 PEFT 模型：名字 → (底座 model_id, 训练时的 max_length)。
# adapter 目录里只有几 MB 的增量权重，底座要单独从 Hub / 缓存加载再套上去——
# 这正是 PEFT 的卖点，但也意味着这里必须记住当初用的是哪个底座。
# Prefix-Tuning 的底座是 roberta-base 而非 DeBERTa，原因见 docs/peft-lora.md。
PEFT_MODELS = {
    "deberta_lora": ("microsoft/deberta-v3-large", 384),
    "deberta_adalora": ("microsoft/deberta-v3-large", 384),
    "deberta_ptuning": ("microsoft/deberta-v3-large", 384),
    "deberta_prefix": ("roberta-base", 384),
}


_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """把两种来源的同一条评论归一到可比形式。

    官方 Kaggle 文件用 `\\"` 转义正文里的引号、整个字段两端还带引号；
    我们从 aclImdb 重建时压平了空白。不做这个归一化，
    直接比字符串会有近一半「对不上」，看起来像是两批不同的数据。
    """
    return _WHITESPACE.sub(" ", str(text).replace('\\"', '"')).strip().strip('"')


def load_test_frame() -> tuple[pd.DataFrame, pd.Series | None]:
    """读测试集。有 testDataWithLabels.tsv 就顺带返回真实标签。"""
    test = pd.read_csv(common.CORPUS_DIR / "testData.tsv", header=0,
                       delimiter="\t", quoting=csv.QUOTE_NONE)
    labelled_path = common.CORPUS_DIR / "testDataWithLabels.tsv"
    if not labelled_path.exists():
        return test, None

    labelled = pd.read_csv(labelled_path, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)

    # 先按 id 对齐。官方 testData.tsv 的 id 形如 "12311_10"（带引号），
    # 和我们从 aclImdb 重建的 0_2 完全不同，这时 id 对不上是预期的。
    merged = test[["id"]].merge(labelled[["id", "sentiment"]], on="id", how="left")
    if not merged["sentiment"].isna().any():
        return test, merged["sentiment"].astype(int)

    # id 对不上就退回按正文对齐。官方文件把内部引号转义成 \"，
    # 而 aclImdb 原文里是裸引号，所以必须先反转义再比——否则近半数会对不上。
    print("  id 与 testDataWithLabels.tsv 不匹配，改按评论正文对齐")
    truth = test[["review"]].assign(key=test["review"].map(_normalize)).merge(
        labelled.assign(key=labelled["review"].map(_normalize))[["key", "sentiment"]]
        .drop_duplicates("key"), on="key", how="left")
    matched = int(truth["sentiment"].notna().sum())
    print(f"  正文对齐成功 {matched:,} / {len(test):,} 条")
    if matched < 0.95 * len(test):
        print("  ⚠ 对齐率过低，跳过本地打分，只生成提交文件")
        return test, None
    return test, truth["sentiment"]


def glove_probabilities(name: str, bundle: common.Bundle, test: pd.DataFrame,
                        device: torch.device) -> np.ndarray:
    module = importlib.import_module(GLOVE_MODELS[name])
    model = module.SentimentNet(bundle.weight)
    common.load_best(name, model, device)

    # 一律按当前 TSV 重新编码，不复用 pickle 里的 test_features。
    #
    # 这一点很容易踩坑：Kaggle 官方 testData.tsv 和我们从 aclImdb 重建的那份
    # **行数都是 25,000，但行顺序完全不同**。如果只用行数是否相等来决定
    # 「复用 pickle」，换成官方文件后就会拿 A 的预测配 B 的 id——
    # 提交文件看起来完全正常，分数却等于随机。重新编码只多花几秒。
    features = common.encode_texts(test["review"].tolist(), bundle.word_to_idx)

    logits = common.predict_logits(model, features, device)
    return torch.softmax(logits, dim=1)[:, 1].numpy()


def hf_probabilities(name: str, test: pd.DataFrame, device: torch.device,
                     batch_size: int = 64, max_length: int = 256) -> np.ndarray:
    from core import hf_trainer  # noqa: F401  —— 顺带关掉 transformers 的 TF 探测
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint = common.MODELS_DIR / f"{name}_hf"
    if not checkpoint.exists():
        raise FileNotFoundError(f"找不到 {checkpoint}，请先跑 python experiments/finetune/{name}.py")

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


def peft_probabilities(name: str, test: pd.DataFrame, device: torch.device,
                       batch_size: int = 32) -> np.ndarray:
    """加载「冻结底座 + adapter」做推理。

    和 `hf_probabilities` 的区别：那边 checkpoint 目录里是一份完整的模型；
    这边目录里只有几 MB 的 adapter，底座得单独加载再套上去。

    LoRA / AdaLoRA 这里其实可以 `merge_and_unload()` 把增量合并回底座，
    推理就完全没有额外开销了。但 P-Tuning / Prefix 无法合并（它们改的是输入，
    不是权重），为了四种方法走同一条代码路径，这里统一不合并——
    差别只有几个百分点的推理耗时。
    """
    from core import peft_trainer  # noqa: F401  —— 顺带关掉 TF 探测、设好 alloc conf
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from core import mem_guard

    checkpoint = common.MODELS_DIR / f"{name}_peft"
    if not checkpoint.exists():
        raise FileNotFoundError(f"找不到 {checkpoint}，请先跑 python experiments/peft/{name.removeprefix("deberta_")}.py")

    base_id, max_length = PEFT_MODELS[name]
    mem_guard.cap_gpu()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    base = AutoModelForSequenceClassification.from_pretrained(base_id, num_labels=2)
    model = PeftModel.from_pretrained(base, str(checkpoint)).to(device).eval()

    texts = test["review"].astype(str).tolist()
    chunks = []
    # 用 fp16 自动混合精度推理，和训练时的口径一致。
    # 走 fp32 的话在 T4 上慢 2~3 倍（这卡的 fp16 tensor core 是 fp32 的数倍算力）——
    # 实测 25,000 条从约 9 分钟涨到约 30 分钟，四个模型就是两小时。
    # softmax 前先 `.float()`，避免在 fp16 里做指数运算丢精度。
    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if device.type == "cuda" else torch.autocast("cpu", enabled=False))
    with torch.no_grad(), autocast:
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], truncation=True,
                              max_length=max_length, padding=True,
                              return_tensors="pt").to(device)
            logits = model(**batch).logits
            chunks.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
    print(f"  底座 {base_id} + adapter，显存峰值 {mem_guard.gpu_peak_gb():.2f} GB")
    return torch.cat(chunks).numpy()


def score_one(name: str, bundle: common.Bundle | None, test: pd.DataFrame,
              truth: pd.Series | None, device: torch.device) -> dict:
    print(f"\n=== {name} ===")
    if name in PEFT_MODELS:
        probabilities = peft_probabilities(name, test, device)
    elif name in HF_MODELS:
        probabilities = hf_probabilities(name, test, device)
    else:
        probabilities = glove_probabilities(name, bundle, test, device)

    predictions = (probabilities >= 0.5).astype(int)
    row = {"model": name}

    if truth is not None:
        # 按正文对齐时会有几十条对不上（aclImdb 原文与官方文件的编码差异），
        # 只在对上的那些行上算指标——把 NaN 一起送进 sklearn 会直接报错。
        mask = truth.notna().to_numpy()
        y_true = truth[mask].astype(int).to_numpy()
        row["scored_rows"] = int(mask.sum())
        row["test_acc"] = round(float(accuracy_score(y_true, predictions[mask])), 4)
        row["test_auc"] = round(float(roc_auc_score(y_true, probabilities[mask])), 4)
        print(f"  测试集准确率 {row['test_acc']:.4f}   ROC-AUC {row['test_auc']:.4f}"
              f"   （在 {row['scored_rows']:,} / {len(test):,} 条有标签的行上）")
    else:
        print("  （无标签，只生成提交文件）")

    # Kaggle 的指标是 AUC，所以提交概率而不是 0/1
    path = common.RESULTS_DIR / f"{name}_submission.csv"
    write_submission(path, test["id"], probabilities)
    print(f"  已写出 {path.name}")
    return row


def write_submission(path: Path, ids: pd.Series, probabilities: np.ndarray) -> None:
    """按 sampleSubmission.csv 的格式逐字节对齐地写出。

    官方样例的标题行是带引号的 `"id","sentiment"`，id 字段本身也带引号
    （我们用 QUOTE_NONE 读入，所以 id 字符串里已经含引号了）。
    pandas 的 to_csv 不会给标题加引号，所以这里手写标题行。
    Kaggle 对标题引号其实是宽容的，但和样例完全一致最省心。
    """
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('"id","sentiment"\n')
        for identifier, probability in zip(ids, probabilities):
            handle.write(f"{identifier},{probability:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="roberta",
                        help="模型名，或 all（全部已训练的模型）/ peft（只跑阶段三的四种）")
    args = parser.parse_args()

    device = common.get_device()
    print(f"设备：{device}")
    test, truth = load_test_frame()
    print(f"测试集 {len(test):,} 条" + ("，含真实标签" if truth is not None else "，无标签"))

    if args.model == "all":
        names = list(GLOVE_MODELS) + list(HF_MODELS) + list(PEFT_MODELS)
    elif args.model == "peft":
        names = list(PEFT_MODELS)
    else:
        names = [args.model]

    rows = []
    for name in names:
        # 只有 GloVe 系模型需要那份 pickle；transformer 系自己带 tokenizer
        needs_bundle = name not in HF_MODELS and name not in PEFT_MODELS
        try:
            rows.append(score_one(name, common.load_data() if needs_bundle else None,
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
