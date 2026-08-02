# 实测结果与分析

所有数字都在同一台机器、同一份数据上跑出，划分方式统一为：25,000 条标注影评
按 8:2 分层划分（`train_test_split(test_size=0.2, random_state=42, stratify=y)`）。

## 汇总

| 方法 | 特征维度 | 特征含义 | Accuracy | ROC-AUC |
|---|--:|---|--:|--:|
| Part 1 · BoW + 随机森林 | 5,000 | 某个具体词的出现次数 | 0.8394 | 0.9140 |
| Part 3-A · Word2Vec 向量平均 + 随机森林 | 300 | 稠密坐标，不可单独解读 | 0.7994 | 0.8782 |
| Part 3-B · Word2Vec 语义簇计数 + 随机森林 | 3,298 | 某个语义簇的词数 | 0.8292 | 0.9070 |
| Part 4 · TF-IDF (1-2gram) + 逻辑回归 | 200,000 | 词/词组的 TF-IDF 权重 | **0.8918** | **0.9576** |

## Part 1 · Bag of Words

```
词表大小: 5,000
特征矩阵: (20000, 5000)，稀疏度 1.3985%
Accuracy: 0.8394   ROC-AUC: 0.9140
```

稀疏度 1.4% 意味着 **98.6% 的元素是 0**。原教程调用 `.toarray()` 把它稠密化，
25,000 × 5,000 的 float64 约 1 GB；保留稀疏格式只需约 11 MB，
随机森林可以直接接受 `scipy.sparse` 输入。

## Part 2 · Word2Vec

```
语料: 25,000 标注 + 50,000 无标注 = 75,000 条影评
分句: 927,909 句，17,800,938 词
词表: 16,493 词（min_count=40 过滤后）× 300 维
训练: CBOW，5 epochs，约 50 秒（8 线程）
```

最近邻检查——这是判断词向量是否真的学到语义最直接的方式：

```
awful      → terrible, atrocious, horrible, dreadful, abysmal, horrendous
brilliant  → superb, fantastic, masterful, terrific, marvelous, wonderful
actress    → actor, performer, comedienne, dancer, role, aishwarya
france     → spain, italy, england, germany, greece, russia
```

四组结果分别体现了：负面情感词聚类、正面情感词聚类、职业角色聚类、国家名聚类。
注意 `actress → actor` 说明模型学到的是「分布相似」而非「意思相同」——
这两个词出现在几乎相同的上下文里。

向量算术：

```
king - man + woman  =  queen (0.581), prince (0.528), princess (0.527)
```

## Part 3 · 文档向量

```
方法 A 向量平均:    (25000, 300)     Accuracy 0.7994   ROC-AUC 0.8782
方法 B 语义簇计数:  (25000, 3298)    Accuracy 0.8292   ROC-AUC 0.9070
```

K-Means 聚类结果抽查（簇数 = 16,493 / 5 = 3,298）：

```
簇    4: ['finds', 'meets', 'sees', 'dies', 'discovers', 'learns', 'realizes']
簇    8: ['isn', 'wasn', 'aren', 'weren']
簇   17: ['print', 'disc', 'edition', 'transfer']
簇   18: ['bond', 'stewart', 'cagney', 'cameron', 'mason', 'garner']
簇    3: ['buffs', 'readers', 'historians']
```

聚类质量很好：簇 4 是第三人称单数动词，簇 8 是被 `[^a-zA-Z]` 清洗截断的缩写否定词
（`isn't → isn`），簇 17 是影碟发行相关，簇 18 是演员姓氏。

**为什么方法 B 明显好于方法 A？**

方法 B 保留了「计数」这个结构：一条影评里出现了 5 个负面情感词，
对应维度就是 5。方法 A 取平均，5 个负面词和 1 个负面词平均出来的方向差别不大——
**强度信息被归一化掉了**。

## Part 4 · TF-IDF + 逻辑回归

```
特征数（unigram + bigram）: 200,000
Accuracy: 0.8918   ROC-AUC: 0.9576
```

加这一节是为了避免一个误导性印象：如果只看 Part 1，容易以为
「稀疏词频特征最多只能到 84%」。事实上同样是稀疏特征，换三处就能到 89%：

1. **TF-IDF 替代原始计数**——降低高频泛用词的权重；
2. **加入 bigram**——`not good` 成为独立特征。纯 unigram 看到的是 `not` 和 `good`
   两个正交维度，否定关系完全丢失；
3. **线性模型替代随机森林**——高维稀疏空间里，决策树每次分裂只看一个维度，
   效率很低；逻辑回归对所有维度加权求和，天然契合这种特征。

## 结论

**Word2Vec 在这个任务上没有赢，但这不构成对 Word2Vec 的否定。**

三个方法都在同一个数据集上做同一个二分类任务。在这种设定下：

- 稀疏特征 + 线性模型是极强的基线，2015 年前后长期难以被超越；
- Word2Vec 的平均聚合丢掉了词序、否定和强度，而这些对情感分类恰恰关键；
- 7.5 万条影评的语料规模，训不出高质量的通用词向量。

Word2Vec 真正的贡献在别处：**它产出的是可迁移的表示**。这份词向量可以拿去做
命名实体识别、文本聚类、相似度检索，而 Part 1 / Part 4 里统计出的词表和
TF-IDF 权重只对这个数据集有效。「预训练一次、下游复用」的范式从这里开始，
一直延续到今天的大模型。

至于「平均会丢信息」这个缺陷，答案是注意力机制——不再等权平均，
而是让模型按相关性动态加权。详见 [从 BoW 到大模型 Embedding](from-bow-to-llm.md)。

## 复现说明

| 部分 | 是否逐位可复现 | 说明 |
|---|:--:|---|
| Part 1 | ✅ | `random_state=42`，多次运行完全一致 |
| Part 2 | ⚠️ | `workers>1` 时 gensim 多线程 SGD 顺序不定；`--workers 1` 可严格复现 |
| Part 3 | ⚠️ | 依赖 Part 2 产物，跟随浮动约 ±0.005 |
| Part 4 | ✅ | 完全一致 |

实测 Part 2 的浮动范围：`king-man+woman → queen` 的相似度在 0.52 ~ 0.58 之间，
但 `queen` 排名第一始终稳定；Part 3-A 的 accuracy 在 0.7964 ~ 0.7994 之间。
这个量级的浮动不影响任何结论。
