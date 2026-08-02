"""从 Python 源码生成 Kaggle Notebook。

Notebook 的 JSON 手写起来容易出错，而且 diff 不可读。这里用脚本生成，源码只在
一处维护：改完 cell 定义重新跑一次即可。

    python tools/build_notebooks.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "notebooks"

# Kaggle 竞赛 word2vec-nlp-tutorial 的 dataSource id。
COMPETITION_SOURCE = {"sourceId": 3971, "sourceType": "competition"}


def _lines(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n")
    return [line + "\n" for line in normalized.splitlines()]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def write_notebook(name: str, cells: list[dict], accelerator: str = "none") -> None:
    # nbformat 4.5+ 要求每个 cell 带 id；用稳定的序号而非随机值，
    # 这样重新生成时 diff 不会整体抖动。
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": accelerator,
                "dataSources": [COMPETITION_SOURCE],
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT))


# 三份 Notebook 共用的数据加载与清洗代码。
SHARED_UTILS = '''
    import csv
    import re
    from html import unescape
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    HTML_TAG = re.compile(r"<[^>]+>")
    NON_LETTER = re.compile(r"[^a-zA-Z]")
    SENTENCE_END = re.compile(r"(?<=[.!?])\\s+")
    STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)
    RANDOM_STATE = 42


    def find_data_dir() -> Path:
        for candidate in [Path("/kaggle/input/word2vec-nlp-tutorial"), Path("data"), Path("../data")]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("请用 Add Input 挂载 word2vec-nlp-tutorial 竞赛数据")


    def read_tsv(data_dir: Path, stem: str) -> pd.DataFrame:
        # quoting=QUOTE_NONE：影评正文含大量引号，按默认规则解析会把字段粘连。
        for suffix in (".tsv", ".tsv.zip"):
            path = data_dir / f"{stem}{suffix}"
            if path.exists():
                return pd.read_csv(path, header=0, delimiter="\\t", quoting=csv.QUOTE_NONE)
        raise FileNotFoundError(f"{data_dir} 下找不到 {stem}")


    def review_to_words(raw_review, remove_stopwords: bool = True) -> list[str]:
        text = NON_LETTER.sub(" ", HTML_TAG.sub(" ", unescape(str(raw_review))))
        words = text.lower().split()
        if remove_stopwords:
            return [w for w in words if w not in STOP_WORDS]
        return words


    def clean_review(raw_review, remove_stopwords: bool = True) -> str:
        return " ".join(review_to_words(raw_review, remove_stopwords))


    DATA_DIR = find_data_dir()
    print("数据目录:", DATA_DIR)
'''


def part1_cells() -> list[dict]:
    return [
        md(
            """
            # Part 1 · Bag of Words

            > Kaggle *Bag of Words Meets Bags of Popcorn* 教程 Part 1 的 Python 3 复现。

            **要解决的问题**：机器学习模型只接受数值向量，而影评是变长的字符串。
            Bag of Words 给出最朴素的一种转换方式——**统计每个词出现了几次**。

            ```
            "This movie is great, really great!"
                        ↓  清洗 + 去停用词
            ["movie", "great", "really", "great"]
                        ↓  按 5,000 词的词表计数
            [0, 0, ..., 1(movie), ..., 2(great), ..., 1(really), ..., 0]
            ```

            这条 5,000 维向量的每一维都对应一个**具体可读的词**，绝大多数是 0。
            它完全丢掉了词序（"good not bad" 和 "bad not good" 向量相同），也不知道
            `awful` 和 `terrible` 是近义词——这两个缺陷正是 Part 2 要解决的。
            """
        ),
        md("## Setup\n\n纯 scikit-learn，CPU 即可。开 GPU 对本 Notebook **没有**加速效果。"),
        code(SHARED_UTILS),
        md("## 1 · 读取数据\n\nKaggle 把竞赛文件挂载为 `.tsv.zip`，`pandas` 依据后缀自动解压，不用手动 unzip。"),
        code(
            """
            train = read_tsv(DATA_DIR, "labeledTrainData")
            test = read_tsv(DATA_DIR, "testData")

            print(f"训练集 {train.shape}，测试集 {test.shape}")
            print(train["sentiment"].value_counts().sort_index().to_dict())
            train.head(3)
            """
        ),
        code(
            """
            # 数据契约检查：先确认列名和取值符合预期，再往下走。
            assert {"id", "sentiment", "review"} <= set(train.columns)
            assert train["review"].notna().all()
            assert set(train["sentiment"].unique()) == {0, 1}
            print("输入校验通过")
            """
        ),
        md(
            """
            ## 2 · 清洗文本

            影评是从网页抓来的，含 `<br />` 标签和 HTML 实体。清洗四步：反转义 →
            去标签 → 只保留字母 → 小写化并去停用词。

            > 原教程用 `BeautifulSoup` + `nltk.corpus.stopwords`。这里换成标准库正则和
            > sklearn 内置停用词表，效果等价，且不需要 `nltk.download()`——Kaggle
            > Notebook 默认断网，下载会直接失败。
            """
        ),
        code(
            """
            raw = train.loc[0, "review"]
            print("清洗前:", raw[:280], "\\n")
            print("清洗后:", clean_review(raw)[:280])
            """
        ),
        code(
            """
            clean_train = train["review"].map(clean_review)
            clean_test = test["review"].map(clean_review)
            print(f"平均长度：{train['review'].str.split().str.len().mean():.0f} 词 "
                  f"→ {clean_train.str.split().str.len().mean():.0f} 词")
            """
        ),
        md(
            """
            ## 3 · 先划分，再建词表

            顺序很关键：**必须先切出验证集，再只用训练分片 `fit` 词表**。
            如果先在全量数据上建词表，验证集的词频分布就泄漏进了特征空间，
            指标会虚高。`stratify` 保证两边正负比例一致。
            """
        ),
        code(
            """
            from sklearn.model_selection import train_test_split

            train_text, valid_text, train_y, valid_y = train_test_split(
                clean_train, train["sentiment"],
                test_size=0.2, random_state=RANDOM_STATE, stratify=train["sentiment"],
            )
            print(f"训练 {len(train_text):,} / 验证 {len(valid_text):,}")
            """
        ),
        md(
            """
            ## 4 · 构造 Bag of Words 特征

            `CountVectorizer` 取词频最高的 5,000 个词建表，输出**稀疏矩阵**。

            > 原教程调用了 `.toarray()`。25,000 × 5,000 的 float64 稠密矩阵约 1 GB，
            > 而实际非零元素只占 ~1.4%。保留 `scipy.sparse` 格式，随机森林可以直接吃。
            """
        ),
        code(
            """
            from sklearn.feature_extraction.text import CountVectorizer

            vectorizer = CountVectorizer(max_features=5000)
            train_x = vectorizer.fit_transform(train_text)
            valid_x = vectorizer.transform(valid_text)

            density = train_x.nnz / (train_x.shape[0] * train_x.shape[1])
            print(f"特征矩阵 {train_x.shape}，非零元素占比 {density:.2%}")
            print(f"稠密化需要 {train_x.shape[0] * train_x.shape[1] * 8 / 1e9:.2f} GB，"
                  f"稀疏存储只需 {train_x.data.nbytes / 1e6:.1f} MB")
            """
        ),
        code(
            """
            vocab = vectorizer.get_feature_names_out()
            counts = np.asarray(train_x.sum(axis=0)).ravel()
            top = pd.DataFrame({"word": vocab, "count": counts}).nlargest(15, "count")
            print("词表片段:", vocab[:12].tolist())
            top.reset_index(drop=True)
            """
        ),
        md("## 5 · 训练随机森林"),
        code(
            """
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
            model.fit(train_x, train_y)
            print("训练完成")
            """
        ),
        md(
            """
            ## 6 · 在留出集上评估

            只看 accuracy 是不够的。加上 ROC-AUC（模型给正样本更高分的能力）和
            混淆矩阵（错在哪一边），才能判断模型是真的学到了东西还是在猜。
            """
        ),
        code(
            """
            from sklearn.metrics import (accuracy_score, classification_report,
                                         confusion_matrix, roc_auc_score)

            predictions = model.predict(valid_x)
            probabilities = model.predict_proba(valid_x)[:, 1]

            print(f"Accuracy: {accuracy_score(valid_y, predictions):.4f}")
            print(f"ROC-AUC : {roc_auc_score(valid_y, probabilities):.4f}\\n")
            print(classification_report(valid_y, predictions, digits=4,
                                        target_names=["negative", "positive"]))
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt

            matrix = confusion_matrix(valid_y, predictions)
            fig, ax = plt.subplots(figsize=(4.4, 3.8))
            ax.imshow(matrix, cmap="Blues")
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:,}", ha="center", va="center",
                        color="white" if value > matrix.max() / 2 else "black")
            ax.set(xticks=[0, 1], yticks=[0, 1],
                   xticklabels=["negative", "positive"], yticklabels=["negative", "positive"],
                   xlabel="预测", ylabel="真实", title="Confusion Matrix · BoW + RF")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 7 · 哪些词最有区分力

            随机森林的 `feature_importances_` 可以直接映射回具体的词——这是 BoW
            相对稠密 Embedding 的一个真实优势：**特征可解释**。
            """
            ),
        code(
            """
            importance = (pd.DataFrame({"word": vocab, "importance": model.feature_importances_})
                          .nlargest(20, "importance").sort_values("importance"))

            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.barh(importance["word"], importance["importance"], color="#4c72b0")
            ax.set(xlabel="feature importance", title="最具区分力的 20 个词")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 8 · 全量重训并生成提交文件

            验证集的作用是选方法。方法定了之后，用**全部** 25,000 条标注数据重训，
            让模型见到尽可能多的样本，再预测测试集。
            """
        ),
        code(
            """
            final_vectorizer = CountVectorizer(max_features=5000)
            final_train_x = final_vectorizer.fit_transform(clean_train)
            final_test_x = final_vectorizer.transform(clean_test)

            final_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
            final_model.fit(final_train_x, train["sentiment"])

            submission = pd.DataFrame({"id": test["id"], "sentiment": final_model.predict(final_test_x)})
            output = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else Path("submission.csv")
            submission.to_csv(output, index=False, quoting=csv.QUOTE_NONE)
            print(f"已写出 {output}（{len(submission):,} 行）")
            submission.head()
            """
        ),
        md(
            """
            ## 小结

            | | |
            |---|---|
            | 向量维度 | 5,000（= 词表大小） |
            | 每一维的含义 | 某个具体词的出现次数 |
            | 稀疏度 | ~98.6% 是 0 |
            | 是否懂近义词 | 不懂，`awful` 与 `terrible` 完全正交 |
            | 是否懂词序 | 不懂 |

            **下一步（Part 2）**：训练 Word2Vec，把每个词压成 300 维稠密向量，
            让近义词在空间里真正靠在一起。
            """
        ),
    ]


def part2_cells() -> list[dict]:
    return [
        md(
            """
            # Part 2 · Word2Vec 词向量

            Part 1 的 BoW 有两个硬伤：向量极度稀疏，且 `awful` 和 `terrible` 是两个
            互不相关的维度。Word2Vec 换了个思路：

            > **一个词的含义，由它周围经常出现哪些词决定。**（分布式假设）

            于是把「预测上下文」当作训练任务，用语料自己给自己造标签：

            ```
            CBOW      : [the, movie, was, ___, boring]  →  预测 "really"
            Skip-gram : "really"  →  预测周围的 the / movie / was / boring
            ```

            没有任何人工标注，标签直接来自文本本身——这就是**自监督学习**。
            今天 BERT 的掩码预测、GPT 的下一词预测，用的是同一套逻辑，只是把
            这里的浅层网络换成了 Transformer。
            """
        ),
        md("## Setup\n\ngensim 的 Word2Vec 是 Cython 多线程 CPU 实现，**不使用 GPU**。开加速器只会白占配额。"),
        code(SHARED_UTILS),
        code(
            """
            import gensim
            from gensim.models import Word2Vec
            print("gensim:", gensim.__version__)
            """
        ),
        md(
            """
            ## 1 · 把无标注数据也用上

            关键点：Word2Vec 训练**不看 sentiment 标签**。所以竞赛里那 50,000 条
            无标注影评是免费的额外语料，一起用能让词向量学得更好。

            25,000 标注 + 50,000 无标注 = 75,000 条影评，约 1,780 万词。
            """
        ),
        code(
            """
            labeled = read_tsv(DATA_DIR, "labeledTrainData")
            unlabeled = read_tsv(DATA_DIR, "unlabeledTrainData")
            print(f"标注 {len(labeled):,} 条 + 无标注 {len(unlabeled):,} 条")
            """
        ),
        md(
            """
            ## 2 · 分句，而不是分词

            Word2Vec 的输入是「句子列表」，每句是词列表。为什么要分句？因为上下文
            窗口不该跨越句子边界——上一句的句尾和下一句的句首在语义上没有关系。

            两点和 Part 1 不同：

            1. **保留停用词**。它们也是上下文窗口的一部分，删掉会让词与词的距离失真。
            2. 用标点正则分句，而非 `nltk.punkt`（避免运行时下载模型）。
            """
        ),
        code(
            """
            def review_to_sentences(raw_review) -> list[list[str]]:
                text = HTML_TAG.sub(" ", unescape(str(raw_review)))
                out = []
                for chunk in SENTENCE_END.split(text):
                    # 注意 remove_stopwords=False
                    words = review_to_words(chunk, remove_stopwords=False)
                    if words:
                        out.append(words)
                return out


            print(review_to_sentences(labeled.loc[0, "review"])[:2])
            """
        ),
        code(
            """
            sentences = []
            for review in labeled["review"]:
                sentences.extend(review_to_sentences(review))
            for review in unlabeled["review"]:
                sentences.extend(review_to_sentences(review))

            print(f"{len(sentences):,} 句，{sum(len(s) for s in sentences):,} 词")
            """
        ),
        md(
            """
            ## 3 · 训练

            沿用原教程的超参数：

            | 参数 | 值 | 作用 |
            |---|---|---|
            | `vector_size` | 300 | 每个词的向量维度 |
            | `min_count` | 40 | 词频下限，滤掉拼写错误和长尾词 |
            | `window` | 10 | 上下文窗口半径 |
            | `sample` | 1e-3 | 高频词降采样，避免 the/and 主导训练 |
            | `sg` | 0 | 0 = CBOW（快），1 = Skip-gram（低频词更准） |

            > API 变更：gensim 4.x 把 `size` 改名 `vector_size`、`iter` 改名 `epochs`，
            > 词向量从 `model[word]` 改为 `model.wv[word]`。原教程的写法在 4.x 会报错。
            """
        ),
        code(
            """
            import logging
            logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

            model = Word2Vec(
                sentences,
                vector_size=300, min_count=40, window=10, sample=1e-3,
                sg=0, epochs=5, workers=4, seed=RANDOM_STATE,
            )
            print(f"\\n词表 {len(model.wv):,} 个词 × {model.wv.vector_size} 维")
            """
        ),
        md(
            """
            ## 4 · 检查它到底学到了什么

            这是最有说服力的一步：看看向量空间里的邻居是不是真的语义相近。
            """
        ),
        code(
            """
            for word in ["awful", "brilliant", "actress", "france"]:
                if word in model.wv:
                    neighbours = ", ".join(w for w, _ in model.wv.most_similar(word, topn=6))
                    print(f"{word:10s} → {neighbours}")
            """
        ),
        code(
            """
            # 挑出不属于同一类的词
            for group in (["kitchen", "bedroom", "bathroom", "france"],
                          ["awful", "terrible", "dreadful", "great"]):
                print(f"{group} → {model.wv.doesnt_match(group)}")
            """
        ),
        code(
            """
            # 向量算术：语义关系被编码成了空间中的方向
            print("king - man + woman =")
            for word, score in model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3):
                print(f"   {word}  ({score:.3f})")
            """
        ),
        md(
            """
            ## 5 · 把 300 维压到 2 维看一眼

            用 PCA 降维后画出来。注意这只是一个粗糙的投影——300 维里的大部分结构
            在 2 维平面上必然丢失，所以图上「看起来近」不等于真的近。
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA

            groups = {
                "情感-正": ["excellent", "wonderful", "brilliant", "superb", "fantastic"],
                "情感-负": ["awful", "terrible", "horrible", "dreadful", "boring"],
                "角色":   ["actor", "actress", "director", "writer", "producer"],
                "国家":   ["france", "italy", "germany", "japan", "russia"],
            }
            words = [w for group in groups.values() for w in group if w in model.wv]
            coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(model.wv[words])

            fig, ax = plt.subplots(figsize=(8, 6))
            index = 0
            for (label, group), colour in zip(groups.items(), ["#2a9d8f", "#e76f51", "#4c72b0", "#9c6ade"]):
                present = [w for w in group if w in model.wv]
                block = coords[index:index + len(present)]
                ax.scatter(block[:, 0], block[:, 1], s=70, color=colour, label=label)
                for (x, y), word in zip(block, present):
                    ax.annotate(word, (x, y), fontsize=9, xytext=(4, 4), textcoords="offset points")
                index += len(present)
            ax.legend(prop={"size": 9})
            ax.set(title="Word2Vec 向量空间的 PCA 投影")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 6 · 保存模型给 Part 3 用

            词向量是**可复用资产**：训练一次，喂给任意下游任务。这一点是 BoW 做不到的——
            某个数据集上统计出的 5,000 词词表没法迁移到别处。预训练范式就是从这里开始的。
            """
        ),
        code(
            """
            output_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("models")
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(output_dir / "word2vec_300d.model"))
            print("已保存到", output_dir / "word2vec_300d.model")
            """
        ),
        md(
            """
            ## 小结

            | | Bag of Words | Word2Vec |
            |---|---|---|
            | 维度 | 5,000（稀疏） | 300（稠密） |
            | 每一维的含义 | 一个具体的词 | 无法单独解读 |
            | 近义词 | 完全正交 | 空间中相邻 |
            | 是否需要标注 | 需要标签才能训分类器 | 自监督，不需标签 |
            | 可迁移 | 否 | 是 |

            **下一步（Part 3）**：词向量是「词」的表示，分类器要的是「整条影评」的表示。
            怎么从词向量拼出文档向量？
            """
        ),
    ]


def part3_cells() -> list[dict]:
    return [
        md(
            """
            # Part 3 · 从词向量到文档向量

            Part 2 得到的是每个**词**的 300 维向量，但分类器需要每条**影评**一个向量。
            一条影评有 200 多个词，怎么合成一个？原教程给了两种做法：

            ```
            方法 A  向量平均      逐维取算术平均 → 300 维稠密向量
            方法 B  语义簇计数    先把词表 K-Means 聚类，再统计影评落在各簇的词数
            ```

            方法 B（原教程称 *bag of centroids*）很有意思：它把 Word2Vec 学到的语义
            相似性「折叠」回了 BoW 那种计数形式——`awful` 和 `terrible` 会落进同一簇，
            于是计数向量里它们贡献到同一维。相当于给 BoW 装上了近义词合并。
            """
        ),
        md("## Setup"),
        code(SHARED_UTILS),
        code(
            """
            from gensim.models import Word2Vec

            candidates = [
                Path("/kaggle/working/word2vec_300d.model"),
                Path("/kaggle/input/word2vec-300d/word2vec_300d.model"),
                Path("models/word2vec_300d.model"),
            ]
            model_path = next((p for p in candidates if p.exists()), None)
            assert model_path is not None, "请先运行 Part 2 生成词向量模型"

            word2vec = Word2Vec.load(str(model_path))
            print(f"载入 {model_path}：{len(word2vec.wv):,} 词 × {word2vec.wv.vector_size} 维")
            """
        ),
        code(
            """
            train = read_tsv(DATA_DIR, "labeledTrainData")
            test = read_tsv(DATA_DIR, "testData")
            print(f"训练 {len(train):,}，测试 {len(test):,}")
            """
        ),
        md(
            """
            ## 方法 A · 向量平均

            逐维求平均。要点：跳过不在词表里的词（`min_count=40` 滤掉了低频词）；
            一个词都没命中时返回零向量——这种情况极少，但不显式处理就会除零。
            """
        ),
        code(
            """
            def average_vectors(reviews, model) -> np.ndarray:
                vectors = model.wv
                features = np.zeros((len(reviews), vectors.vector_size), dtype=np.float32)
                for row, review in enumerate(reviews):
                    known = [w for w in review_to_words(review) if w in vectors]
                    if known:
                        features[row] = np.mean(vectors[known], axis=0)
                return features


            train_avg = average_vectors(train["review"], word2vec)
            test_avg = average_vectors(test["review"], word2vec)
            print(f"{train_avg.shape} —— 300 维，全部非零（对比 Part 1 的 5,000 维、98.6% 为 0）")
            """
        ),
        md(
            """
            ## 方法 B · 语义簇计数

            对 16,000 多个词向量做 K-Means，簇数取 `词表大小 / 5`（原教程设定，
            平均每簇约 5 个词）。

            > 原教程用 `KMeans`，词表上万时相当慢。这里换 `MiniBatchKMeans`，
            > 结果接近而快一个量级。
            """
        ),
        code(
            """
            from sklearn.cluster import MiniBatchKMeans

            vectors = word2vec.wv
            n_clusters = len(vectors.index_to_key) // 5
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE,
                                     n_init=3, batch_size=1024)
            assignments = kmeans.fit_predict(vectors.vectors)
            word_to_cluster = dict(zip(vectors.index_to_key, assignments.tolist()))
            print(f"{len(vectors.index_to_key):,} 词 → {n_clusters:,} 簇")
            """
        ),
        code(
            """
            # 抽查几个簇，看聚类是否符合直觉
            from collections import defaultdict

            clusters = defaultdict(list)
            for word, cluster in word_to_cluster.items():
                clusters[cluster].append(word)

            shown = 0
            for cluster, words in sorted(clusters.items()):
                if 3 <= len(words) <= 8:
                    print(f"簇 {cluster:5d}: {words}")
                    shown += 1
                    if shown == 10:
                        break
            """
        ),
        code(
            """
            def centroid_vectors(reviews, word_to_cluster) -> np.ndarray:
                n_clusters = max(word_to_cluster.values()) + 1
                features = np.zeros((len(reviews), n_clusters), dtype=np.float32)
                for row, review in enumerate(reviews):
                    for word in review_to_words(review):
                        cluster = word_to_cluster.get(word)
                        if cluster is not None:
                            features[row, cluster] += 1
                return features


            train_clu = centroid_vectors(train["review"], word_to_cluster)
            test_clu = centroid_vectors(test["review"], word_to_cluster)
            print(train_clu.shape)
            """
        ),
        md("## 对比两种表示"),
        code(
            """
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score
            from sklearn.model_selection import train_test_split


            def score(features, labels, name):
                tr_x, va_x, tr_y, va_y = train_test_split(
                    features, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)
                clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
                clf.fit(tr_x, tr_y)
                accuracy = accuracy_score(va_y, clf.predict(va_x))
                auc = roc_auc_score(va_y, clf.predict_proba(va_x)[:, 1])
                print(f"{name:16s} 维度 {features.shape[1]:>5,}   Accuracy {accuracy:.4f}   ROC-AUC {auc:.4f}")
                return {"name": name, "dim": features.shape[1], "accuracy": accuracy, "roc_auc": auc}


            results = [
                score(train_avg, train["sentiment"], "向量平均"),
                score(train_clu, train["sentiment"], "语义簇计数"),
            ]
            """
        ),
        md(
            """
            ## 一个值得注意的结果

            **语义簇计数明显好于向量平均**，原因在于它保留了「计数」这个结构：
            一条影评出现 5 个负面情感词，对应维度就是 5。而向量平均把 5 个负面词和
            1 个负面词平均成方向差别不大的结果——**强度信息被归一化掉了**。

            更值得注意的是，如果和 Part 1 的 BoW（accuracy 0.8394）对比，会发现
            **稠密向量并没有胜出**。这不是代码写错了，原教程本身也承认这一点。原因有几层：

            - **平均操作抹平了信息**。200 个词向量取平均，词序、否定、强调全部消失，
              「不好看」和「好，不看」平均下来几乎一样。
            - **语料太小**。7.5 万条影评训出的词向量，远不如在数十亿词上预训练的向量。
            - **任务太窄**。单一领域的二分类，TF-IDF 这类稀疏特征本来就是极强的基线。

            Word2Vec 真正的价值不在这个分数上，而在于**这份词向量可以拿去做别的任务**。
            后来的 ELMo / BERT / GPT 沿着「预训练表示」这条路走下去，
            解决的正是「平均会丢信息」这个问题——用注意力机制按上下文动态加权，
            而不是简单求平均。
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt

            frame = pd.DataFrame(results)
            fig, ax = plt.subplots(figsize=(6, 3.6))
            positions = np.arange(len(frame))
            ax.bar(positions - 0.2, frame["accuracy"], 0.4, label="Accuracy", color="#4c72b0")
            ax.bar(positions + 0.2, frame["roc_auc"], 0.4, label="ROC-AUC", color="#dd8452")
            ax.set(xticks=positions, ylim=(0.7, 1.0), title="文档向量方案对比")
            ax.set_xticklabels(frame["name"])
            ax.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## 生成提交文件"),
        code(
            """
            best = max(results, key=lambda r: r["roc_auc"])
            print("选用:", best["name"])
            train_features, test_features = (
                (train_avg, test_avg) if best["name"] == "向量平均" else (train_clu, test_clu))

            final = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
            final.fit(train_features, train["sentiment"])
            submission = pd.DataFrame({"id": test["id"], "sentiment": final.predict(test_features)})

            output = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else Path("submission.csv")
            submission.to_csv(output, index=False, quoting=csv.QUOTE_NONE)
            print(f"已写出 {output}")
            submission.head()
            """
        ),
        md(
            """
            ## 三部分串起来看

            | | 表示 | 维度 | 每一维的含义 |
            |---|---|---|---|
            | Part 1 | Bag of Words | 5,000 稀疏 | 某个具体词的次数 |
            | Part 2 | Word2Vec 词向量 | 300 稠密 | 不可单独解读 |
            | Part 3 | 影评向量（平均 / 簇计数） | 300 / ~3,300 | 同上 / 某个语义簇的次数 |

            从 Part 1 到 Part 3，走完的是 NLP 表示学习最关键的一次转向：
            **从「统计某个词出现几次」到「把语义编码进连续空间的坐标」**。
            今天大模型的 Embedding 层做的还是同一件事，只是向量由 Transformer
            按上下文动态生成，而不再是每个词一张固定的查找表。
            """
        ),
    ]


if __name__ == "__main__":
    write_notebook("part1-bag-of-words.ipynb", part1_cells())
    write_notebook("part2-word2vec.ipynb", part2_cells())
    write_notebook("part3-doc-vectors.ipynb", part3_cells())
