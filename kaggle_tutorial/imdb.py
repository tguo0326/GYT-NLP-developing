"""数据加载与文本清洗：Part 1 / 2 / 3 共用的基础工具。

原教程用 `BeautifulSoup` 去 HTML、用 `nltk` 取停用词，两者都要额外下载资源
（`nltk.download("stopwords")` 在无网络的 Kaggle Notebook 里会失败）。这里改为
标准库 `html.unescape` + 正则，以及 scikit-learn 自带的英文停用词表，行为等价
但零外部下载。
"""

from __future__ import annotations

import csv
import re
from html import unescape
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 竞赛数据在不同环境下的挂载位置：Kaggle Notebook / 本地 data/ / 旧版相对路径。
SEARCH_DIRS = (
    Path("/kaggle/input/word2vec-nlp-tutorial"),
    Path("data"),
    Path("../data"),
    Path("../input/word2vec-nlp-tutorial"),
)

HTML_TAG = re.compile(r"<[^>]+>")
NON_LETTER = re.compile(r"[^a-zA-Z]")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)


def find_data_dir(explicit: Path | None = None) -> Path:
    """定位竞赛数据目录；显式传入的路径优先。"""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"数据目录不存在：{explicit}")
        return explicit
    for candidate in SEARCH_DIRS:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in SEARCH_DIRS)
    raise FileNotFoundError(
        "找不到竞赛数据。已尝试：" + searched + "\n"
        "在 Kaggle 上请用 Add Input 挂载 word2vec-nlp-tutorial；"
        "在本地可先运行 tools/make_local_dataset.py。"
    )


def read_tsv(data_dir: Path, stem: str) -> pd.DataFrame:
    """读取 `<stem>.tsv` 或 Kaggle 挂载的 `<stem>.tsv.zip`。

    `quoting=csv.QUOTE_NONE` 是必须的：影评正文里大量出现英文引号，若按默认
    规则解析引号，字段会被错误合并。pandas 会根据 `.zip` 后缀自动解压。
    """
    for suffix in (".tsv", ".tsv.zip"):
        path = data_dir / f"{stem}{suffix}"
        if path.exists():
            return pd.read_csv(path, header=0, delimiter="\t", quoting=csv.QUOTE_NONE)
    raise FileNotFoundError(f"{data_dir} 下找不到 {stem}.tsv 或 {stem}.tsv.zip")


def review_to_words(raw_review: object, remove_stopwords: bool = True) -> list[str]:
    """把一条原始影评转成小写词列表。

    步骤：HTML 反转义 → 去标签 → 非字母字符替换为空格 → 转小写切词 → 可选去停用词。

    Part 1 需要去停用词（降低词频特征维度）；Part 2 训练 Word2Vec 时建议保留，
    因为停用词也构成上下文窗口的一部分，删掉会打乱词序距离。
    """
    text = NON_LETTER.sub(" ", HTML_TAG.sub(" ", unescape(str(raw_review))))
    words = text.lower().split()
    if remove_stopwords:
        return [word for word in words if word not in STOP_WORDS]
    return words


def clean_review(raw_review: object, remove_stopwords: bool = True) -> str:
    """`review_to_words` 的字符串版本，供 CountVectorizer / TfidfVectorizer 使用。"""
    return " ".join(review_to_words(raw_review, remove_stopwords))


def review_to_sentences(raw_review: object) -> list[list[str]]:
    """把一条影评切成句子，每句再切成词——Word2Vec 的训练输入格式。

    原教程使用 `nltk.punkt` 分句器。为了避免运行时下载模型，这里改用标点正则。
    对影评这种口语化文本，两者在句子边界上的差异对词向量质量影响很小。
    """
    text = HTML_TAG.sub(" ", unescape(str(raw_review)))
    sentences = []
    for raw_sentence in SENTENCE_END.split(text):
        words = review_to_words(raw_sentence, remove_stopwords=False)
        if words:
            sentences.append(words)
    return sentences
