# 从 Bag of Words 到大模型 Embedding

这份笔记回答一个问题：**这个 2014 年的 Kaggle 教程，和今天的大模型 Embedding
是什么关系？**

答案是：它们解决的是**同一个问题**——把离散的语言符号变成连续向量——但方案的
表达能力差了几代。教程里的三种方法，恰好是这条演化路径上的前三站。

---

## 一、所有方法都在回答同一个问题

计算机不能直接处理「电影很好看」这五个字。任何 NLP 系统的第一步都是：

```
离散符号序列  ──表示层──▶  实数向量  ──计算层──▶  预测
```

区别只在「表示层」怎么做。

## 二、第一站：One-Hot 与 Bag of Words

**One-Hot**：词表有 V 个词，第 i 个词就是一个 V 维向量，第 i 位是 1，其余为 0。

```
movie  = [0, 0, 1, 0, 0, ..., 0]   ← 5,000 维
film   = [0, 1, 0, 0, 0, ..., 0]
```

问题一眼可见：`movie` 和 `film` 的余弦相似度是 **0**。任意两个不同的词都彼此正交，
向量空间里不存在「语义距离」这个概念。

**Bag of Words** 是把一句话里所有词的 One-Hot 加起来（即词频统计）：

```
"good movie, really good"  →  [..., 2(good), ..., 1(movie), ..., 1(really), ...]
```

它的三个根本局限：

| 局限 | 表现 |
|---|---|
| 维度灾难 | 维度 = 词表大小，动辄数万；98%+ 是 0 |
| 语义盲 | 近义词完全正交，无法泛化 |
| 词序丢失 | "good not bad" 与 "bad not good" 向量相同 |

但它有一个真实优势：**每一维都对应一个具体的词，模型完全可解释**。
Part 1 的随机森林能直接告出「哪些词最有区分力」，这是稠密向量做不到的。

## 三、第二站：Word2Vec —— 分布式表示

2013 年 Mikolov 等人的 Word2Vec 换了个前提假设：

> **一个词的含义，由它经常出现在什么样的上下文中决定。**
> （分布式假设，Distributional Hypothesis，语言学界 1950 年代就已提出）

于是把「预测上下文」构造成一个训练任务：

```
CBOW      (Continuous Bag of Words)
    [the, movie, was, ___, boring]  →  预测中心词 "really"

Skip-gram
    "really"  →  预测周围的 the / movie / was / boring
```

**关键洞察：这个任务的标签不需要人工标注。** 输入和标签都从语料自身切出来，
语料本身就是监督信号。这就是**自监督学习**（self-supervised learning）。

训练完成后，隐层权重矩阵的每一行就是一个词的稠密向量：

```
movie  = [ 0.23, -1.04,  0.87, ..., 0.12]   ← 300 维，全部非零
film   = [ 0.21, -0.98,  0.91, ..., 0.09]   ← 和 movie 很接近
```

三个质变：

1. **维度从 5,000 降到 300**，且信息密度大幅提高；
2. **近义词在空间中相邻**——本项目实测 `awful` 的最近邻是
   `terrible, atrocious, horrible, dreadful, abysmal`；
3. **语义关系被编码成空间中的方向**——本项目实测
   `king - man + woman ≈ queen (0.581)`。

这些不是理论推导，是本仓库 `src/part2_word2vec.py` 在 7.5 万条影评上跑出的
实际结果。可以自己跑一遍验证。

## 四、Word2Vec 的天花板

教程 Part 3 揭示了一个反直觉的结果：**把词向量平均成文档向量后，分类效果
并没有超过 Part 1 的 BoW。** 本项目实测数字见主 README 的对比表。

原因不在实现，而在方法本身：

**局限 1：静态向量，一词一义。**
`bank` 在「河岸」和「银行」两个意思下共用同一个向量。Word2Vec 的向量是查找表里
一行固定的数，不随上下文改变。

**局限 2：聚合会丢信息。**
把 200 个词向量取平均，词序、否定、强调全部被抹平。
「这片子不好看」和「好，这片子不看」平均下来几乎一样。

**局限 3：窗口有限。**
window=10 只能看到局部上下文，跨句、跨段落的依赖完全捕捉不到。

## 五、第三站：上下文相关的 Embedding

后续研究基本都在攻这三个局限：

| 年份 | 模型 | 关键突破 |
|---|---|---|
| 2013 | Word2Vec | 稠密向量 + 自监督预训练 |
| 2014 | GloVe | 全局共现统计与局部窗口结合 |
| 2016 | FastText | 字符 n-gram，能处理未登录词 |
| 2018 | ELMo | **双向 LSTM 生成随上下文变化的向量** |
| 2018 | BERT | Transformer 编码器 + 掩码语言建模，深度双向 |
| 2018→ | GPT 系列 | Transformer 解码器 + 下一词预测，规模化 |

**ELMo 的分水岭意义**：向量不再是查找表里的固定行，而是模型读完整句后**算**出来的。
`bank` 在「river bank」和「bank account」里得到不同向量——局限 1 解决了。

**Transformer 的自注意力**则解决了局限 2 和 3：不再简单平均，而是让每个词按
相关性动态加权聚合全序列的信息。序列多长都能直接连边，不受窗口限制。

## 六、今天大模型的 Embedding 层

打开任何一个现代 LLM，第一层还是一张 Embedding 查找表：

```python
# 概念示意
self.embed_tokens = nn.Embedding(vocab_size, hidden_size)   # 128k × 4096
```

**这一层的机制和 Word2Vec 完全一致**：一个词（准确说是一个 token）对应向量空间
里的一行，随训练更新。

但整个系统有三处本质不同：

| | Word2Vec | 大模型 |
|---|---|---|
| 切分单位 | 完整单词 | BPE / SentencePiece **子词** |
| 词表大小 | 1.6 万（本项目） | 10 万 ~ 20 万 |
| 向量维度 | 300 | 4,096 ~ 16,384 |
| 训练语料 | 1,780 万词 | 数万亿 token |
| 是否随上下文变化 | ❌ 查表即最终结果 | ✅ 经数十层注意力后逐层重算 |
| 训练目标 | 预测窗口内的上下文 | 预测下一个 token（+ 后训练对齐） |

最关键的是最后两行：Embedding 层只是**入口**。真正被下游使用的「句子表示」是
经过几十层 Transformer 之后的隐状态——那才是上下文相关的表示。
Word2Vec 相当于只有入口，没有后面的楼层。

而共同点也很实在：

- **都是自监督预训练。** 从 Word2Vec 的窗口预测到 GPT 的下一词预测，一脉相承。
- **都把语义编码为连续空间的几何关系。** 相似度用余弦距离衡量，
  向量算术在两者上都成立（大模型上更强）。
- **都产出可迁移的表示。** 这是相对 BoW 最本质的进步——
  BoW 的词表绑死在某个数据集上，无法迁移。

## 七、一句话总结这条路

```
One-Hot         正交，无语义
   ↓  统计
Bag of Words    可解释，但稀疏、盲于语义、丢词序
   ↓  自监督预测上下文
Word2Vec        稠密、有语义，但一词一义、聚合丢信息
   ↓  用整句上下文动态生成
ELMo / BERT     同一个词在不同语境下不同向量
   ↓  规模化 + 注意力堆深
大模型 Embedding  子词粒度、超高维、逐层重算的上下文表示
```

回到最初的问题：**「目前大模型的 Embedding 也是这么来做的」这个说法对不对？**

- 对的部分：核心思想同源——把词映射到稠密向量、用自监督任务从无标注语料学习、
  用几何距离表达语义相似。Word2Vec 确立的正是这套范式。
- 不完全对的部分：Bag of Words **不是** Embedding，它是稀疏计数特征。
  而大模型的 Embedding 层虽然形式上就是查找表，但真正起作用的上下文表示
  是几十层注意力算出来的，跟 Word2Vec 的静态查表已经隔了好几代。

准确的说法是：**Word2Vec 是现代 Embedding 的直接祖先；Bag of Words 是它的前史。**

## 参考文献

- Mikolov et al., 2013. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Mikolov et al., 2013. [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- Pennington et al., 2014. [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/)
- Bojanowski et al., 2016. [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)（FastText）
- Peters et al., 2018. [Deep Contextualized Word Representations](https://arxiv.org/abs/1802.05365)（ELMo）
- Vaswani et al., 2017. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Devlin et al., 2018. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- Maas et al., 2011. [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015/)（IMDB 数据集原论文）
