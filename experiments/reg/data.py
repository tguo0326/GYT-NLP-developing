"""IMDB (Kaggle word2vec-nlp-tutorial) 数据加载与切分。

三个训练脚本共用这里，避免每个文件复制一遍读 tsv 的代码。
"""
import logging
import os

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# 语料默认取仓库根的 corpus/（和 core/common.py 一致），不入库，用 tools/ 重建
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.environ.get("IMDB_DIR", os.path.join(_ROOT, "corpus", "imdb"))
# 提交文件和 metrics 都落到这里，和 submissions/ 下其他阶段一个格式。
# 从 __file__ 推导而不是写相对路径，这样从仓库根跑 `python experiments/reg/xxx.py` 也对。
SUBMISSION_DIR = os.path.join(_ROOT, "submissions", "17_rdrop_scl")
# 训练中间产物，.gitignore 里 models/ 已经挡住了
MODELS_DIR = os.path.join(_ROOT, "models")


def load_raw(data_dir=DEFAULT_DATA_DIR):
    """读原始 tsv。quoting=3 = QUOTE_NONE，影评里有裸引号，不关掉会串行。"""
    train_path = os.path.join(data_dir, "labeledTrainData.tsv")
    test_path = os.path.join(data_dir, "testData.tsv")
    for path in (train_path, test_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} 不存在。用环境变量 IMDB_DIR 指定数据目录，"
                f"或把 Kaggle word2vec-nlp-tutorial 的 tsv 放到 {data_dir}/")
    train = pd.read_csv(train_path, header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(test_path, header=0, delimiter="\t", quoting=3)
    # quoting=3 会把 id 外层的引号也当字面字符读进来（'"12311_10"'），
    # Kaggle 能收，但本地跟带标签的测试集对不上，统一剥掉。
    for df in (train, test):
        if "id" in df.columns:
            df["id"] = df["id"].astype(str).str.strip('"')
    return train, test


def build_datasets(tokenizer, data_dir=DEFAULT_DATA_DIR, max_length=512,
                   val_size=0.2, seed=3407, limit=None):
    """返回 (train_ds, val_ds, test_ds, test_ids)。

    标签列统一叫 `labels`，这是 HuggingFace Trainer / 模型 forward 的约定名，
    改成别的名字（比如 `label`）会导致 loss 拿不到标签。
    limit 只用于本地 smoke test，正式实验必须留 None。
    """
    train_df, test_df = load_raw(data_dir)
    train_df, val_df = train_test_split(train_df, test_size=val_size,
                                        random_state=seed,
                                        stratify=train_df["sentiment"])

    if limit is not None:
        logger.warning("limit=%s，只用了子集，结果不可用于汇报", limit)
        train_df, val_df, test_df = train_df[:limit], val_df[:limit], test_df[:limit]

    train_ds = Dataset.from_dict({"labels": train_df["sentiment"].tolist(),
                                  "text": train_df["review"].tolist()})
    val_ds = Dataset.from_dict({"labels": val_df["sentiment"].tolist(),
                                "text": val_df["review"].tolist()})
    test_ds = Dataset.from_dict({"text": test_df["review"].tolist()})

    def tokenize(examples):
        return tokenizer(examples["text"], max_length=max_length, truncation=True)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    logger.info("train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))
    return train_ds, val_ds, test_ds, test_df["id"].tolist()


def save_submission(test_ids, preds, path):
    """写 Kaggle 提交文件，列名固定为 id / sentiment。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame({"id": test_ids, "sentiment": preds}).to_csv(
        path, index=False, quoting=3)
    logger.info("submission saved to %s", path)
